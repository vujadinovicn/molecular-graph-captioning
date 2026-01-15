import os
import argparse  
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.collater import MolecularCaptioningCollator
from data.dataset import MolecularCaptioningDataset
from losses.loss import InfoNCELoss
from models.model import MolecularCaptioningModel
from models.graph_encoder import GINEEncoder
from data.utils import get_num_embeddings_list
from models.utils import save_contrastive_model_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train Contrastive Model Phase (Graph + Projector)")

    # Paths
    parser.add_argument("--graphs_path", type=str, required=True, 
                        help="Path to the directory containing graph data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--llm_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help="HuggingFace model ID for the LLM")

    # Hyperparameters
    parser.add_argument("--temperature", type=float, default=0.15, help="Temperature for contrastive loss")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for Graph Encoder and Projector")
    parser.add_argument("--graph_encoder_lr", type=float, default=2e-4, help="Learning rate for Projector parameters")
    parser.add_argument("--projector_lr", type=float, default=2e-4, help="Learning rate for Projector parameters")
    
    # Logging & Saving Intervals
    parser.add_argument("--log_interval", type=int, default=50, help="Print loss every N steps")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save mid-epoch checkpoint every N steps")

    return parser.parse_args()

@torch.no_grad()
def full_eval(model, val_loader, temperature=0.15, device="cuda"):
    model.eval()

    all_graph_embs, all_text_embs = [], []
    for batch in val_loader:
        batch = batch.to(device)
        graph_emb, text_emb = model.forward_contrastive(batch, readout_fn="mean")
        all_graph_embs.append(graph_emb.float())
        all_text_embs.append(text_emb.float())

    all_graph_embs = torch.cat(all_graph_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)

    def metrics(query, key):
        sims = (query @ key.t()) / temperature
        ranks = sims.argsort(dim=-1, descending=True)
        N = query.size(0)
        correct = torch.arange(N, device=sims.device)
        pos = (ranks == correct.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1

        out = {"MRR": (1.0 / pos.float()).mean().item()}
        for k in [1, 5, 10]:
            hitk = (pos <= k).float().mean().item()
            out[f"R@{k}"] = hitk
            out[f"Hit@{k}"] = hitk
        return out

    results = {
        "t2g": metrics(all_text_embs, all_graph_embs),
        "g2t": metrics(all_graph_embs, all_text_embs),
    }

    model.train()
    return results

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

    train_dataset = MolecularCaptioningDataset(
        graphs_path=args.graphs_path,
        split="train",
        tokenizer=tokenizer,
    )
    
    train_collator = MolecularCaptioningCollator(tokenizer, mode="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,          
        shuffle=True,          
        collate_fn=train_collator,    
        num_workers=args.num_workers,          
        pin_memory=True         
    )

    val_dataset = MolecularCaptioningDataset(
        graphs_path=args.graphs_path,
        split="validation",
        tokenizer=tokenizer,
    )

    val_collator = MolecularCaptioningCollator(tokenizer, mode="validation")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,          
        shuffle=False,          
        collate_fn=val_collator,    
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

    model = MolecularCaptioningModel(
        graph_encoder=graph_encoder,
        llm_model=base_llm,
        tokenizer=tokenizer,
        node_dim=args.hidden_dim,
    )
    
    # Freeze LLM 
    for param in model.llm.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.graph_encoder.to(device)
    model.node_projector.to(device)

    # Parameter Grouping
    projector_params = []
    graph_encoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "projector" in name:
            projector_params.append(param)
        elif "graph_encoder" in name:
            graph_encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': graph_encoder_params, 'lr': args.graph_encoder_lr},          
        {'params': projector_params, 'lr': args.projector_lr}
    ])

    # 4. Training Loop
    model.train()
    loss_fn = InfoNCELoss(temperature=args.temperature)

    loss_over_batches = 0
    best_mrr = float("-inf")
    best_epoch = -1

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):  
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            batch["graph_batch"] = batch["graph_batch"].to(device)
            batch["description_input_ids"] = batch["description_input_ids"].to(device)
            batch["description_attention_mask"] = batch["description_attention_mask"].to(device)
            
            optimizer.zero_grad()

            print("here")
            
            # graph_embs, text_embs = model.forward_contrastive(batch, readout_fn="mix")
            # loss = loss_fn(graph_embs, text_embs)
            # loss_over_batches += loss.item()

            # loss.backward()
            # optimizer.step()

            loss = torch.tensor(0.0)  # Placeholder to avoid errors
            
            if batch_idx % args.log_interval == 0 and batch_idx:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss_over_batches:.4f}")
                loss_over_batches = 0
                
                if batch_idx % args.checkpoint_interval == 0:
                    mid_path = os.path.join(args.save_dir, "contrastive_learning_model_mid.pth")
                    save_contrastive_model_checkpoint(model, mid_path)
        
        # Save if best mrr at end of epoch
        metrics = full_eval(model, val_loader, temperature=loss_fn.temperature, device=device)
        print(f"Validation Epoch {epoch} | MRR={metrics['mrr']:.4f} | g2t={metrics['g2t']} | t2g={metrics['t2g']}")
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_path = os.path.join(args.save_dir, "best_mrr.pth")
            save_contrastive_model_checkpoint(model, best_path)
            print(f"Saved best MRR checkpoint!")

        # Save every 5 epochs epoch
        if (epoch + 1) % 5 == 0:
            epoch_path = os.path.join(args.save_dir, f"contrastive_learning_model_epoch_{epoch}.pth")
            save_contrastive_model_checkpoint(model, epoch_path)

    print("Training complete.")