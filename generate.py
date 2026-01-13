import argparse
import json
import pickle
import torch
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Custom imports (Assuming these exist in your environment)
from load_data import MolecularCaptioningDataset, MultimodalCollator
from model import GINEEncoder, MoleculeLlama3
from utils import get_num_embeddings_list

def parse_args():
    parser = argparse.ArgumentParser(description="Molecular Captioning Inference Script")

    # --- Paths ---
    parser.add_argument("--main_ckpt_path", type=str, required=True, 
                        help="Path to the main fine-tuned model checkpoint (e.g., ft_freeze_graph_llama_1B_epoch_4.pth)")
    parser.add_argument("--graph_ckpt_path", type=str, required=True,
                        help="Path to the graph encoder specific checkpoint (e.g., contrast_mix_24.pth)")
    parser.add_argument("--data_path", type=str, default="/home/shishirk/adityasr/kshitij_molecular_captioning/data_baseline/data",
                        help="Path to the root directory of the graph data")
    parser.add_argument("--output_file", type=str, default="new_graph_BEAm_plus.json",
                        help="Path to save the output JSON")

    # --- Configuration ---
    parser.add_argument("--llm_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace Model ID for the base LLM")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for Graph Encoder")

    return parser.parse_args()

def load_model_checkpoint(model, ckpt_path):
    """Loads model state dict from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {ckpt_path}")
    return model

@torch.no_grad()
def generate_caption(model, tokenizer, graph_batch, device):
    """
    Generates captions for a batch of graphs.
    Generation parameters are HARDCODED here.
    """
    model.eval()
    
    graph_batch = graph_batch.to(device)
    batch_size = graph_batch.num_graphs if hasattr(graph_batch, 'num_graphs') else graph_batch.batch_size
    
    # Encode graph
    node_emb, node_mask, _ = model.graph_encoder(graph_batch)
    
    # Project to LLM Dim
    proj_nodes = model.node_projector(node_emb)     
    graph_embeds = proj_nodes # [Batch, Max_Nodes, LLM_Dim]
    
    # Graph Mask
    graph_full_mask = node_mask # Shape: [Batch, Max_Nodes]

    num_graph_tokens = graph_embeds.shape[1] 
    special_token = "<|reserved_special_token_1|>"
       
    fixed_system_message = (
            "You are a helpful assistant that captions molecules based on their structure. "
            "Provide concise and informative descriptions. Following are a few examples of captions:"
            "## Example 1: The molecule is a 4-O-[(E)-2-methyl-2-butenoyl]ascaroside derived from (2E,8R)-8-hydroxynon-2-enoic acid. It is a metabolite of the nematode Caenorhabditis elegans. It has a role as a Caenorhabditis elegans metabolite. It is a 4-O-[(E)-2-methyl-2-butenoyl]ascaroside and an alpha,beta-unsaturated monocarboxylic acid. It derives from an ascr#3 and a (2E,8R)-8-hydroxynon-2-enoic acid."
            "## Example 2: The molecule is an alkanesulfonic acid in which the alkyl group directly linked to the sulfo functionality is methyl. It has a role as an Escherichia coli metabolite. It is an alkanesulfonic acid and a one-carbon compound. It is a conjugate acid of a methanesulfonate."
        )
    user_message_content = f"Caption the following molecule: {special_token * num_graph_tokens}"

    conversation = [
        {"role": "system", "content": fixed_system_message},
        {"role": "user", "content": user_message_content},
    ]
    
    # Tokenize ONCE, then expand for the batch
    input_ids = tokenizer.apply_chat_template(
        conversation, 
        add_generation_prompt=True,
        tokenize=True, 
        return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Expand to match batch size [Batch, Seq_Len]
    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = attention_mask.repeat(batch_size, 1)
    
    # Get base text embeddings [Batch, Seq_Len, LLM_Dim]
    inputs_embeds = model.llm.get_input_embeddings()(input_ids)

    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
    replacement_mask = (input_ids == special_token_id)
    
    # Replace special tokens with graph embeddings
    inputs_embeds[replacement_mask] = graph_embeds.reshape(-1, graph_embeds.shape[-1]).to(inputs_embeds.dtype)
    attention_mask[replacement_mask] = graph_full_mask.reshape(-1).to(attention_mask.dtype)

    # --- HARDCODED GENERATION PARAMETERS ---
    gen_kwargs = dict(
        do_sample=False,
        num_beams=8,                  
        num_beam_groups=4,            
        diversity_penalty=0.3,        
        num_return_sequences=1,       
        length_penalty=1.15,          
        early_stopping=False,          
        min_new_tokens=10,            
        max_new_tokens=192,             
        repetition_penalty=1.08,       
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    # ---------------------------------------

    outputs = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **gen_kwargs
    )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id)
    tokenizer.pad_token = tokenizer.eos_token 

    my_custom_token = "<|reserved_special_token_1|>"
    tokenizer.add_tokens([my_custom_token], special_tokens=True)
    special_token_id = tokenizer.convert_tokens_to_ids(my_custom_token)
    print(f"Token: {my_custom_token} | ID: {special_token_id}")

    # 2. Initialize Models
    ATOM_NUM_EMBEDDINGS_LIST, BOND_NUM_EMBEDDINGS_LIST = get_num_embeddings_list()
    
    graph_encoder = GINEEncoder(
        atom_num_embeddings_list=ATOM_NUM_EMBEDDINGS_LIST,
        bond_num_embeddings_list=BOND_NUM_EMBEDDINGS_LIST,
        hidden_dim=args.hidden_dim,
    ) 

    base_llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_llm.resize_token_embeddings(len(tokenizer))

    model = MoleculeLlama3(
        graph_encoder=graph_encoder,
        llm_model=base_llm,
        tokenizer=tokenizer,
        node_dim=args.hidden_dim,
    )

    # 3. Load Checkpoints
    # Load Main Model
    model = load_model_checkpoint(model, args.main_ckpt_path)
    
    # Load Graph Encoder specific checkpoint
    print(f"Loading graph encoder state from {args.graph_ckpt_path}")
    # Note: Loading with map_location to ensure it goes to the right device if needed
    ckpt = torch.load(args.graph_ckpt_path, map_location=device)
    model.graph_encoder.load_state_dict(ckpt["graph_encoder"])

    model.to(device)
    model.eval()

    # 4. Data Loading
    dataset = MolecularCaptioningDataset(
        graphs_path=args.data_path,
        split=args.split,
        tokenizer=tokenizer,
    )
    
    collator = MultimodalCollator(tokenizer)
    inference_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator
    )

    results = []
    print(f"Starting generation for {len(dataset)} examples...")

    # 5. Main Loop
    with torch.no_grad():
        for n, batch in tqdm(enumerate(inference_loader), total=len(inference_loader)):
            
            graph_batch = batch["graph_batch"]
            
            # Run Generation (Parameters are inside the function)
            generated_text = generate_caption(
                model=model,
                tokenizer=tokenizer,
                graph_batch=graph_batch,
                device=device
            ) 

            current_batch_size = len(generated_text)
            
            for i in range(current_batch_size):
                gen_text = generated_text[i]
                
                # Extract ground truth
                labels = batch["labels"][i]
                valid_label_ids = labels[labels != -100]
                ground_truth_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
                
                # Extract IDs with fallback logic
                if "id" in batch:
                    current_id = batch["id"][i]
                elif "graph_id" in batch:
                    current_id = batch["graph_id"][i]
                elif hasattr(graph_batch, "id"): 
                    if isinstance(graph_batch.id, list):
                        current_id = graph_batch.id[i]
                    else:
                        current_id = graph_batch.id[i].item() 
                else:
                    current_id = f"unknown_{n}_{i}" 

                results.append({
                    "id": str(current_id),
                    "generated": gen_text,
                    "ground_truth": ground_truth_text
                })

            if (n+1) % 10 == 0:
                print("-" * 50)
                print(f"Batch {n+1} Example:")
                print(f"Generated: {generated_text[0]}")
                print(f"Truth:     {results[-current_batch_size]['ground_truth']}")
                print("-" * 50)

    # 6. Save to JSON
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nProcessing Complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()

'''
python /home/shishirk/adityasr/kshitij_molecular_captioning/molecular-graph-captioning/generate.py \
  --main_ckpt_path "/home/shishirk/adityasr/kshitij_molecular_captioning/multimodal-prompt-tuning/saved_model/CURRENT_train_1b_instr_nem_grafr/ft_freeze_graph_llama_1B_epoch_4.pth" \
  --graph_ckpt_path "/home/shishirk/adityasr/kshitij_molecular_captioning/nemanja_saved_model/contrast_mix_24.pth" \
  --data_path "/home/shishirk/adityasr/kshitij_molecular_captioning/data_baseline/data" \
  --output_file "results.json"
'''