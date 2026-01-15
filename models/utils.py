import torch 

def load_model_checkpoint(model, ckpt_path):
    """Loads model state dict from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {ckpt_path}")
    return model

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

def save_contrastive_model_checkpoint(model, path):
    ckpt = {
        "graph_encoder": model.graph_encoder.state_dict(),
        "node_projector": model.node_projector.state_dict(),
    }

    torch.save(ckpt, path)
    print(f"Saved contrastive model checkpoint to {path}!")