import torch
from utils.utils import parse_config
import torch
from transformers import AutoTokenizer
from data.dataloader import get_dataloader
from data.dataset import get_dataset
from models.loaders import get_molecular_captioning_model
from losses.info_nce import InfoNCELoss

def train(config):
    graphs_path = config['data'].get('graphs_path', 'data/graphs')
    max_description_length = config['data'].get('max_description_length', 512)
    batch_size = config['train_contrastive'].get('batch_size', 1)
    lr = config['train_contrastive'].get('lr', 1e-4)
    temperature = config['train_contrastive'].get('temperature', 0.3)
    accum_steps = config["train_contrastive"].get("accum_steps", 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    train_dataset = get_dataset(graphs_path, "train", tokenizer, max_description_length)
    train_loader = get_dataloader(train_dataset, "train", batch_size)

    model = get_molecular_captioning_model(config, device)
    # TODO: Nemanja. (Kshitij) Check if it should be symmetric
    loss_fn = InfoNCELoss(temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        graph_embs, text_embs = model.forward_contrastive(batch)
        optimizer.zero_grad()

        graph_embs, text_embs = model.forward_contrastive(batch)

        loss = loss_fn(graph_embs, text_embs) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        break
        
        # node_embs, node_mask, embs = model.graph_encoder(batch)
        # print(node_embs.shape)
        # print(node_mask[0])

        # graph_final_embs = adapter(node_embs)
        # print(graph_final_embs.shape)


if __name__ == "__main__":
    config = parse_config("config/config.yml")
    train(config)

