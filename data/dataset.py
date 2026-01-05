import pickle
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.graphormer.collating_graphormer import preprocess_item
import pandas as pd

class MolGraphormerDataset(Dataset):
    """
    Dataset that loads pre-saved molecule graphs with optional text embeddings.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        emb_dict: Dictionary mapping ID to text embedding tensors (optional)
    """
    def __init__(self, graph_path: str, emb_dict: Dict[str, torch.Tensor] = None):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.emb_dict = emb_dict
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        graph_item = pyg_to_graphormer_item(graph)
        if self.emb_dict is not None:
            id_ = graph.id
            text_emb = self.emb_dict[id_]
            return graph_item, text_emb
        else:
            return graph_item

def pyg_to_graphormer_item(data):
    """
    Convert a PyTorch Geometric Data object to Graphormer input format
    and run Graphormer preprocessing.
    """
    node_feat = data.x.to(torch.long)
    edge_feat = data.edge_attr.to(torch.long)
    edge_index = data.edge_index.to(torch.long)

    item = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": edge_feat,
        "num_nodes": data.num_nodes,
    }

    item = preprocess_item(item)
    return item

def collate_fn(collator, batch):
    """
    Collate function for DataLoader to batch graphs with optional text embeddings.
    
    Args:
        batch: List of graph Data objects or (graph, text_embedding) tuples
        
    Returns:
        Batched graph or (batched_graph, stacked_text_embeddings)
    """
    if isinstance(batch[0], tuple):
        graph_items, text_embs = zip(*batch)
        graph_batch = collator(list(graph_items))
        text_embs = torch.stack(text_embs, dim=0)
        return graph_batch, text_embs
    else:
        return collator(batch)
    
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb
    

if __name__ == "__main__":
    # generic test
    train_emb = load_id2emb("data/bert_embs/train.csv")

    train_ds = MolGraphormerDataset("data/graphs/train_graphs.pkl", train_emb)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(train_dl[0].shape)
    print("el")