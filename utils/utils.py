import yaml
import torch
import pickle

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_pickle(path):
    return pickle.load(open(path, "rb"))

def load_model_checkpoint(model, ckpt_path):
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
        print("Checkpoint is empty!")
    else:
        torch.save(checkpoint, path)
        print(f"Saved model checkpoint to {path}!")

def save_contrastive_checkpoint(model, path):
    ckpt = {
        "graph_encoder": model.graph_encoder.state_dict(),
        "node_projector": model.node_projector.state_dict(),
        "feat1_projector": model.feat1_projector.state_dict(),
        "feat2_projector": model.feat2_projector.state_dict(),
        "use_global_feats": model.use_global_feats
    }

    torch.save(ckpt, path)
    print(f"Saved contrastive model checkpoint to {path}!")

