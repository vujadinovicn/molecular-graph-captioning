import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_num_embeddings_list
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
import random
import pickle
import os
import argparse  
from torch_geometric.data import Batch
from load_data import MolecularCaptioningDataset, MultimodalCollator
from model import GINEEncoder, MoleculeLlama3, AtomBondEncoder, Projector

def save_model_checkpoint(model, path):
    checkpoint = {
        k: v for k, v in model.named_parameters() 
        if v.requires_grad
    }
    if not checkpoint:
        print("WARNING: Checkpoint is empty! No trainable parameters found.")
    else:
        torch.save(checkpoint, path)
        print(f"Saved model checkpoint to {path} (Size: {len(checkpoint)} tensors)")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Molecular Captioning Model (Graph + Llama)")

    # Paths
    parser.add_argument("--graphs_path", type=str, required=True, 
                        help="Path to the directory containing graph data")
    parser.add_argument("--load_checkpoint_path", type=str, required=True, 
                        help="Path to the pre-trained graph encoder/projector checkpoint (.pth)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--llm_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help="HuggingFace model ID for the LLM")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for Graph Encoder and Projector")
    parser.add_argument("--llm_lr", type=float, default=5e-5, help="Learning rate for LoRA/LLM parameters")
    parser.add_argument("--projector_lr", type=float, default=1e-3, help="Learning rate for Projector parameters")
    
    # Logging & Saving Intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Print loss every N steps")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save mid-epoch checkpoint every N steps")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created checkpoint directory: {args.save_dir}")

    ATOM_NUM_EMBEDDINGS_LIST, BOND_NUM_EMBEDDINGS_LIST = get_num_embeddings_list()
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    my_custom_token = "<|reserved_special_token_1|>"
    num_added_toks = tokenizer.add_tokens([my_custom_token], special_tokens=True)

    special_token_id = tokenizer.convert_tokens_to_ids(my_custom_token)
    print(f"Token: {my_custom_token} | ID: {special_token_id}")

    dataset = MolecularCaptioningDataset(
        graphs_path=args.graphs_path,
        split="train",
        tokenizer=tokenizer,
    )
    
    collator = MultimodalCollator(tokenizer)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,          
        shuffle=True,          
        collate_fn=collator,    
        num_workers=args.num_workers,          
        pin_memory=True         
    )

    graph_encoder = GINEEncoder(
        atom_num_embeddings_list=ATOM_NUM_EMBEDDINGS_LIST,
        bond_num_embeddings_list=BOND_NUM_EMBEDDINGS_LIST,
        hidden_dim=args.hidden_dim,
    ) 

    base_llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_llm.resize_token_embeddings(len(tokenizer))

    model = MoleculeLlama3(
        graph_encoder=graph_encoder,
        llm_model=base_llm,
        tokenizer=tokenizer,
        node_dim=args.hidden_dim,
    )

    # Loading from Checkpoint
    print(f"Loading checkpoint from: {args.load_checkpoint_path}")
    ckpt = torch.load(args.load_checkpoint_path, map_location="cuda")
    model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    model.node_projector.load_state_dict(ckpt["node_projector"])

    # Freeze graph encoder
    for param in model.graph_encoder.parameters():
        param.requires_grad = False
    
    # Unfreeze projector
    for param in model.node_projector.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.graph_encoder.to(device)
    model.node_projector.to(device)

    # Parameter Grouping
    lora_params = []
    projector_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "projector" in name:
            projector_params.append(param)
        else:
            lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.llm_lr},          
        {'params': projector_params, 'lr': args.projector_lr}
    ])

    # 4. Training Loop
    model.train()
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):  
        for batch_idx, batch in enumerate(train_loader):
            batch["graph_batch"] = batch["graph_batch"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["labels"] = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0 and batch_idx:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")
                
                if batch_idx % args.checkpoint_interval == 0:
                    mid_path = os.path.join(args.save_dir, "ft_freeze_graph_llama_1B_mid.pth")
                    save_model_checkpoint(model, mid_path)
        
        # Save at end of epoch
        epoch_path = os.path.join(args.save_dir, f"ft_freeze_graph_llama_1B_epoch_{epoch}.pth")
        save_model_checkpoint(model, epoch_path)

    print("Training complete.")