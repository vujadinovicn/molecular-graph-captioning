import torch
from utils.utils import save_contrastive_checkpoint, parse_config
import torch
from transformers import AutoTokenizer
from data.dataloader import get_dataloader
from data.dataset import get_dataset
from models.loaders import get_molecular_captioning_model
from losses.info_nce import InfoNCELoss
import os


def train(config):
    graphs_path = config['data'].get('graphs_path', 'data/graphs')
    max_description_length = config['data'].get('max_description_length', 512)
    batch_size = config['train_contrastive'].get('batch_size', 1)
    # lr = config['train_contrastive'].get('lr', 1e-4)
    temperature = config['train_contrastive'].get('temperature', 0.3)
    accum_steps = config["train_contrastive"].get("accum_steps", 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_molecular_captioning_model(config, device)
    model.train()
    model.freeze_llm()

    train_dataset = get_dataset(graphs_path, "train", tokenizer, max_description_length=max_description_length) # do we specify this for lama?
    train_dataloader = get_dataloader(train_dataset, "train", batch_size)

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
        {'params': graph_encoder_params, 'lr': 1e-3},
        {'params': projector_params, 'lr': 1e-3}
    ])

    # TODO: Nemanja. (Kshitij) Check if it should be symmetric
    loss_fn = InfoNCELoss(temperature)

    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):  
        for batch_idx, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            graph_embs, text_embs = model.forward_contrastive(batch)
            
            optimizer.zero_grad()
            loss = loss_fn(graph_embs, text_embs)
            loss.backward()
            optimizer.step()

            # loss = loss_fn(graph_embs, text_embs) / accum_steps
            # loss.backward()

            # if (step + 1) % accum_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Step {batch_idx} | Loss: {loss.item():.4f}")
                if batch_idx % 500 == 0:
                    os.makedirs("saved_model", exist_ok=True)
                    save_contrastive_checkpoint(model, f"saved_model/contrast.pth")

        save_contrastive_checkpoint(model, f"saved_model/contrast_{epoch}.pth")
    print("Training complete.")

if __name__ == "__main__":
    config = parse_config("config/config.yml")
    train(config)

