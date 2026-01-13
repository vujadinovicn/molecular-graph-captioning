import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.utils.data
import torch_geometric
import torch_geometric.data
import torch_geometric.loader.dataloader

class PyGMolecularCaptioningCollator(torch_geometric.loader.dataloader.Collater):  
    def __init__(self, dataset, tokenizer, mode="train", **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        data_batch = torch_geometric.data.Batch.from_data_list(
            batch, 
            exclude_keys=[
                "prompt_input_ids", 
                "description_input_ids", 
            ]
        )

        text_batch = build_text_batch(batch, self.text_pad_token_id, self.mode)
        data_batch.update(text_batch)

        if self.exclude_keys: 
            data_batch = {
                k: v for k, v in data_batch.items() 
                if k not in self.exclude_keys
            }

        return data_batch
    

def build_text_batch(batch, text_pad_token_id, mode):
    prompt_input_ids = [data["prompt_input_ids"][0] for data in batch]
    pad_prompt_input_ids = pad_sequence(
        prompt_input_ids, 
        padding_value=text_pad_token_id, 
        padding_side="left"
    )
    pad_prompt_attention_mask = pad_sequence(
        [torch.ones_like(data["prompt_input_ids"][0]) for data in batch], 
        padding_value=0, 
        padding_side="left"
    )

    # TODO: Kshitij
    if mode == "test":
        return {
            "input_ids": pad_prompt_input_ids, 
            "attention_mask": pad_prompt_attention_mask, 
        } 

    description_input_ids = [data["description_input_ids"][0] for data in batch]
    pad_description_input_ids = pad_sequence(
        description_input_ids, 
        padding_value=text_pad_token_id, 
        padding_side="right"
    )
    
    pad_description_attention_mask = pad_sequence(
        [torch.ones_like(data["description_input_ids"][0]) for data in batch], 
        padding_value=0, 
        padding_side="right"
    )
    pad_labels = pad_sequence(
        description_input_ids, 
        padding_value=-100,
        padding_side="right"
    )
    
    return {
        "input_ids": torch.cat([pad_prompt_input_ids, pad_description_input_ids], dim=1), 
        "attention_mask": torch.cat([pad_prompt_attention_mask, pad_description_attention_mask], dim=1), 
        "labels": torch.cat([torch.full_like(pad_prompt_input_ids, fill_value=-100), pad_labels], dim=1),
        "description_input_ids": pad_description_input_ids,
        "description_attention_mask": pad_description_attention_mask,
    }
    
def pad_sequence(sequences, padding_value, padding_side="right"):
    max_len = max(sequence.shape[-1] for sequence in sequences)
    padded_sequences = []
    for sequence in sequences: 
        padding = torch.full(
            size=(max_len - sequence.shape[-1],),
            fill_value=padding_value,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        if padding_side == "left":
            padded_sequences.append(torch.cat([padding, sequence], dim=-1))
        elif padding_side == "right":
            padded_sequences.append(torch.cat([sequence, padding], dim=-1))
    return torch.stack(padded_sequences, dim=0)


def get_dataloader(dataset, mode, batch_size):
    return MolecularCaptioningDataLoader(
        dataset=dataset,
        mode=mode,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

class MolecularCaptioningDataLoader(DataLoader): 
    def __init__(self, dataset, mode, batch_size, shuffle, follow_batch=[], exclude_keys=[], **kwargs):
        kwargs.pop("collate_fn", None)

        collate_fn = PyGMolecularCaptioningCollator(
            dataset=dataset,
            tokenizer=dataset.description_tokenizer,
            mode=mode,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )

    def __len__(self):
        return super().__len__()

    def __iter__(self):
        return super().__iter__()
    
def get_dataloader(dataset, mode, batch_size):
    return MolecularCaptioningDataLoader(
        dataset=dataset,
        mode=mode,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


import pickle
from torch.utils.data import Dataset
import torch.utils.data
import os

class MolecularCaptioningDataset(Dataset):
    def __init__(
            self, 
            graphs_path, 
            split="train",
            description_tokenizer=None, 
            max_description_length=512, 
            **kwargs
    ):
        self.graphs_path = graphs_path
        self.description_tokenizer = description_tokenizer
        self.max_description_length = max_description_length
        self.split = split

        self.load_graphs()

    def load_graphs(self):
        with open(os.path.join(self.graphs_path, self.split+"_graphs.pkl"), 'rb') as f:
            self.graphs = pickle.load(f)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        chat = self.build_and_tokenize_chat_prompt(graph)
        description = self.tokenize_description(graph)
        
        graph.id = graph.id
        graph.prompt_input_ids = chat["prompt_input_ids"]
        graph.description_input_ids = description["description_input_ids"]
        return graph

    def build_and_tokenize_chat_prompt(self, graph):
        # TODO: Write the system message and change user message's placeholder_token
        system_message = "Caption the molecule."
        placeholder_token: str = '<|reserved_special_token_1|>'
        num_nodes = graph.num_nodes # we can change this
        user_message = ("Molecule graph embeddings: " + placeholder_token * (num_nodes + 2))
        
        prompt =  [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        prompt_input_ids = self.description_tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            return_tensors="pt"
        )      

        return {"prompt_input_ids": prompt_input_ids}
     
    def tokenize_description(self, graph):
        description = getattr(graph, "description", None)

        if not description:
            return {"description_input_ids": None}

        input_ids = self.description_tokenizer(
            [description],
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        if input_ids.size(-1) > self.max_description_length:
            input_ids = input_ids[:, :self.max_description_length]
            description = self.description_tokenizer.decode(input_ids[0], skip_special_tokens=True)

        description_ids = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]

        return {"description_input_ids": description_ids}
    


def get_dataset(graphs_path, split, tokenizer, max_description_length):
    return MolecularCaptioningDataset(
        graphs_path=graphs_path,
        split=split,
        description_tokenizer=tokenizer,
        max_description_length=max_description_length
    )


x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def get_num_embeddings_list():
    atom_num_embeddings_list = [len(x_map[k]) for k in x_map.keys()]
    bond_num_embeddings_list = [len(e_map[k]) for k in e_map.keys()]
    return atom_num_embeddings_list, bond_num_embeddings_list


@torch.no_grad()
def full_eval(model, val_loader, temperature, device, k_list=(1,5,10)):
    model.eval()

    g2t_correct = {k: 0 for k in k_list}
    t2g_correct = {k: 0 for k in k_list}
    total = 0

    mrr_g2t_sum = 0.0
    mrr_t2g_sum = 0.0

    mean_rank_g2t_sum = 0.0
    mean_rank_t2g_sum = 0.0

    for batch in val_loader:
        batch = batch.to(device)
        G, T = model.forward_contrastive(batch, readout_fn="mean")

        logits = (G.float() @ T.float().T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        sorted_idx = logits.argsort(dim=1, descending=True)
        sorted_idx_t = logits.argsort(dim=0, descending=True)

        # Recall@K
        for k in k_list:
            g2t_correct[k] += (sorted_idx[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
            t2g_correct[k] += (sorted_idx_t[:k, :] == labels.unsqueeze(0)).any(dim=0).sum().item()

        # ranks g2t
        ranks_g2t = (sorted_idx == labels.unsqueeze(1)).nonzero()[:, 1] + 1
        mrr_g2t_sum += (1.0 / ranks_g2t.float()).mean().item()
        mean_rank_g2t_sum += ranks_g2t.float().mean().item()

        # ranks t2g
        ranks_t2g = (sorted_idx_t == labels.unsqueeze(0)).nonzero()[:, 0] + 1
        mrr_t2g_sum += (1.0 / ranks_t2g.float()).mean().item()
        mean_rank_t2g_sum += ranks_t2g.float().mean().item()

        total += logits.size(0)

    g2t_acc = {k: g2t_correct[k] / total for k in k_list}
    t2g_acc = {k: t2g_correct[k] / total for k in k_list}

    num_batches = len(val_loader)
    mrr_g2t = mrr_g2t_sum / num_batches
    mrr_t2g = mrr_t2g_sum / num_batches
    mean_rank_g2t = mean_rank_g2t_sum / num_batches
    mean_rank_t2g = mean_rank_t2g_sum / num_batches

    mrr = 0.5 * (mrr_g2t + mrr_t2g)
    mean_rank = 0.5 * (mean_rank_g2t + mean_rank_t2g)

    model.train()
    return {
        "g2t": g2t_acc,
        "t2g": t2g_acc,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "mrr_g2t": mrr_g2t,
        "mrr_t2g": mrr_t2g,
    }

def plot_similarity_matrix(G, T, title="", tau=0.07, normalize_logits=True, save_path=None, show=False):

    # normalize embeddings (for cosine similarity)
    dtype = torch.float16
    G = G.float()
    T = T.float()
    # similarity matrix
    sim = G @ T.T  # [B, B]
    # sim = T @ T.T
    logits = sim

    if normalize_logits:
        print(logits.min())
        print(logits.mean())
        logits = (logits - logits.mean()) / (logits.std() + 1e-8)

    # plot
    plt.figure(figsize=(6, 5))
    plt.imshow(logits.detach().cpu(), aspect="auto")
    plt.colorbar(label="Normalized logits" if normalize_logits else "Logits")
    plt.title(title)
    plt.xlabel("Text Embedding Sequences")
    plt.ylabel("Graph Embedding Sequences")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    # ✅ Show or not
    if show:
        plt.show()
    else:
        plt.close()



yaml_text = """
data:
  graphs_path: '/home/shishirk/adityasr/kshitij_molecular_captioning/data_baseline/data'
  graph_features: '/home/shishirk/adityasr/kshitij_molecular_captioning/multimodal-prompt-tuning/extracted_features/test_features_dict.pkl'
  max_description_length: 512

models:
 llm_name: 'meta-llama/Llama-3.2-1B-Instruct'
 checkpoint_path: '/home/shishirk/adityasr/kshitij_molecular_captioning/multimodal-prompt-tuning/saved_model/train/full_ft_molecule_llama3_1B_peft_epoch_4.pth'

train_contrastive:
  batch_size: 256
  graph_hidden_dim: 512
  graph_out_dim: 512
  adapter_hidden_dim: 1024
  llama_hidden_dim: 2048
  lr: 1e-4
  temperature: 0.1
  accum_steps: 8

train_instruct:
  batch_size: 64

generation:
  batch_size: 1

train_instruct:
  dummy: 0

eval:
  batch_size: 64
"""

config = yaml.safe_load(yaml_text)

def readout_embeddings(embeddings, attention_mask, readout_fn):
    if readout_fn == "last":
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        return embeddings[batch_indices, last_token_indices, :]

    elif readout_fn == "mean":
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if attention_mask.sum(dim=1, keepdim=True).any() == 0:
          print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO MEAN")
        return sum_embeddings / count_attn_mask
    
    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="mean")
        diff_embeddings = embeddings - mean_embeddings.unsqueeze(1)
        diff_embeddings_2 = diff_embeddings.pow(2) 
        masked_diff_embeddings_2 = diff_embeddings_2 * attention_mask.unsqueeze(-1)
        sum_diff_embeddings_2 = masked_diff_embeddings_2.sum(dim=1) 
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)
        var = sum_diff_embeddings_2 / count_attn_mask
        return torch.sqrt(var + 1e-8)

    elif readout_fn == "mix": 
        mean_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="mean")
        std_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="std")
        return torch.cat([mean_embeddings, std_embeddings], dim=1)
    

class Projector(nn.Module):
    """Projects features from source_dim to llm_dim."""
    def __init__(self, source_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.net(x)

class MolecularCaptioningModel(nn.Module):
    def __init__(
        self,
        graph_encoder,
        llm_model,
        tokenizer,
        node_dim=256,
        global_feat_1_dim=20,
        global_feat_2_dim=1024,
        use_global_feats=False,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.llm = llm_model
        self.tokenizer = tokenizer

        self.llm_dim = llm_model.config.hidden_size

        self.node_projector = Projector(node_dim, self.llm_dim)
        self.feat1_projector = Projector(global_feat_1_dim, self.llm_dim)
        self.feat2_projector = Projector(global_feat_2_dim, self.llm_dim)
        self.use_global_feats = use_global_feats

        self.special_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_1|>")


    def get_graph_embeddings(self, graph_batch, readout_fn, normalize=True):
        node_emb, node_mask, _ = self.graph_encoder(graph_batch)
        proj_tokens = self.node_projector(node_emb)

        if self.use_global_feats:
            print("here")
            g1 = graph_batch.global_feat_1
            g2 = graph_batch.global_feat_2
            proj_g1 = self.feat1_projector(g1).unsqueeze(1)
            proj_g2 = self.feat2_projector(g2).unsqueeze(1)

            all_tokens = torch.cat([proj_g1, proj_g2, proj_tokens], dim=1)
            g1g2_mask = torch.ones(node_mask.size(0), 2, device=node_mask.device, dtype=node_mask.dtype)
            all_mask = torch.cat([g1g2_mask, node_mask], dim=1)

            mean_all = readout_embeddings(all_tokens, all_mask, "mean")
            std_nodes = readout_embeddings(proj_tokens, node_mask, "std")

            embeddings = torch.cat([mean_all, std_nodes], dim=-1)
        else:
            embeddings = readout_embeddings(proj_tokens, node_mask, readout_fn)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def get_description_embeddings(self, input_ids, attention_mask, layer, readout_fn, normalize=True):
        hidden_states = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        ).hidden_states[layer]

        embeddings = readout_embeddings(hidden_states, attention_mask, readout_fn)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward_contrastive(
            self,
            batch,
            text_layer=16,
            readout_fn="mix"
        ):
        graph_emb = self.get_graph_embeddings(
            graph_batch=batch,
            readout_fn=readout_fn,
            normalize=True
        )

        text_emb = self.get_description_embeddings(
            input_ids=batch.description_input_ids,
            attention_mask=batch.description_attention_mask,
            layer=text_layer,
            readout_fn=readout_fn,
            normalize=True
        )

        return graph_emb, text_emb

    def freeze_llm(self):
            for param in self.llm.parameters():
                param.requires_grad = False

from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

class AtomBondEncoder(nn.Module):
    def __init__(self, num_embeddings_list, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, emb_dim)
            for num_embeddings in num_embeddings_list
        ])

    def forward(self, x):
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out

class GINEEncoder(nn.Module):
    def __init__(
        self, 
        atom_num_embeddings_list,
        bond_num_embeddings_list,
        hidden_dim=256,
        out_dim=512,
        num_layers=5,
        dropout=0.1,
        pooling="mean",  # "mean" or "add"
        normalize=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.normalize = normalize

        self.atom_encoder = AtomBondEncoder(atom_num_embeddings_list, hidden_dim)
        self.bond_encoder = AtomBondEncoder(bond_num_embeddings_list, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # mlp = nn.Sequential(
            #     nn.Linear(hidden_dim, hidden_dim // 2),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim // 2, hidden_dim // 2),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim // 2, hidden_dim // 2),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim // 2, hidden_dim),
            # )
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            conv = GINEConv(nn=mlp, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr)

        # for conv, norm in zip(self.convs, self.norms):
        for layer in range(self.num_layers):
            conv, norm = self.convs[layer], self.norms[layer]
            x = conv(x, data.edge_index, e)
            x = norm(x)
            if layer == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)

        if self.pooling == "mean":
            g = global_mean_pool(x, data.batch)
        else:
            g = global_add_pool(x, data.batch)

        g_proj = self.proj(g)
        if self.normalize:
            g_proj = F.normalize(g_proj, dim=-1)

        node_emb, node_mask = to_dense_batch(x, batch=data.batch)
        return node_emb, node_mask, g_proj # return per node embeddings, mask and mean/added embeddings


def get_graph_encoder(config, device):
    graph_hidden_dim = config['train_contrastive'].get('graph_hidden_dim', 512)
    graph_out_dim = config['train_contrastive'].get('graph_out_dim', 512)

    atom_num_embeddings_list, bond_num_embeddings_list = get_num_embeddings_list()
    model = GINEEncoder(
        atom_num_embeddings_list,
        bond_num_embeddings_list,
        hidden_dim=graph_hidden_dim,
        out_dim=graph_out_dim,
        pooling="mean",
    ).to(device)

    return model

def get_llm(config):
    tokenizer = AutoTokenizer.from_pretrained(config['models']['llm_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|reserved_special_token_1|>"], special_tokens=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
            config['models']['llm_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    llm_model.resize_token_embeddings(len(tokenizer))

    return llm_model, tokenizer

def get_molecular_captioning_model(config, device):
    graph_encoder = get_graph_encoder(config, device)
    # modality_adapter = get_modality_adapter(config, device)
    llm_model, tokenizer = get_llm(config)

    model = MolecularCaptioningModel(
        graph_encoder=graph_encoder,
        llm_model=llm_model,
        tokenizer=tokenizer,
        node_dim=config['train_contrastive'].get('graph_out_dim', 512),
        global_feat_1_dim=20,
        global_feat_2_dim=1024,
        use_global_feats=False
    ).to(device)

    return model, tokenizer

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_output, text_output):
        graph_output = graph_output.float()
        text_output = text_output.float()

        logits = (graph_output @ text_output.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_g2t = F.cross_entropy(logits, labels)  # graph->text
        loss_t2g = F.cross_entropy(logits.T, labels)  # text->graph

        loss = (loss_g2t + loss_t2g) / 2
        return loss



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



def train(config, run_with_two_losses=True):
    graphs_path = config['data'].get('graphs_path', 'data/graphs')
    max_description_length = config['data'].get('max_description_length', 512)
    batch_size = config['train_contrastive'].get('batch_size', 1)
    # lr = config['train_contrastive'].get('lr', 1e-4)
    temperature = config['train_contrastive'].get('temperature', 0.1)
    accum_steps = config["train_contrastive"].get("accum_steps", 8)
    accum_steps=4
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
        {'params': graph_encoder_params, 'lr': 2e-4},
        {'params': projector_params, 'lr': 2e-4}
    ])

    temperature = 0.15

    # TODO: Nemanja. (Kshitij) Check if it should be symmetric
    loss_fn = InfoNCELoss(temperature)

    NUM_EPOCHS = 25
    accum_steps = 4
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch = batch.to(device)
            optimizer.zero_grad()

            if run_with_two_losses:
                graph_embs_mean, text_embs_mean = model.forward_contrastive(batch, readout_fn="mean")
                graph_embs_std, text_embs_std = model.forward_contrastive(batch, readout_fn="std")
                loss = 0.3*loss_fn(graph_embs_mean, text_embs_mean) + 0.7*loss_fn(graph_embs_std, text_embs_std)
            else:
                graph_embs_mean, text_embs_mean = model.forward_contrastive(batch, readout_fn="mean")
                loss = loss_fn(graph_embs_mean, text_embs_mean)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % accum_steps == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | BigLoss: {loss.item():.4f}")

        os.makedirs("nemanja_saved_model", exist_ok=True)
        # if epoch+1 % 5 == 0 and epoch > 0:
        if run_with_two_losses:
            save_contrastive_checkpoint(model, f"/home/shishirk/adityasr/kshitij_molecular_captioning/nemanja_saved_model/contrast_mean_std_{epoch}.pth")
        else:
            save_contrastive_checkpoint(model, f"/home/shishirk/adityasr/kshitij_molecular_captioning/nemanja_saved_model/contrast_mix_{epoch}.pth")
    print("Training complete.")

def get_dataloader2(dataset, mode, batch_size):
    return MolecularCaptioningDataLoader(
        dataset=dataset,
        mode=mode,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


if __name__ == "__main__":
    # run_with_two_losses=False
    # train(config, run_with_two_losses)

    #  full_eval(model, val_loader, temperature, device, k_list=(1,5,10)):
    

    graphs_path = config['data'].get('graphs_path', 'data/graphs')
    max_description_length = config['data'].get('max_description_length', 512)
    batch_size = config['train_contrastive'].get('batch_size', 1)
    # lr = config['train_contrastive'].get('lr', 1e-4)
    temperature = config['train_contrastive'].get('temperature', 0.3)
    accum_steps = config["train_contrastive"].get("accum_steps", 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=128

    model, tokenizer = get_molecular_captioning_model(config, device)
    model.eval()
    ckpt = torch.load("/home/shishirk/adityasr/kshitij_molecular_captioning/nemanja_saved_model/contrast_mix_24.pth", map_location="cuda")

    model.graph_encoder.load_state_dict(ckpt["graph_encoder"])
    model.node_projector.load_state_dict(ckpt["node_projector"])
    model.freeze_llm()

    train_dataset = get_dataset(graphs_path, "validation", tokenizer, max_description_length=max_description_length) # do we specify this for lama?
    train_dataloader = get_dataloader2(train_dataset, "validation", batch_size)

    results = full_eval(model, train_dataloader, temperature=0.15, device="cuda", k_list=(1,5,10))
    print(results)
    for batch_idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        G, T = model.forward_contrastive(batch, readout_fn="mean", text_layer=16)
        if batch_idx == 0:
            break

    plot_similarity_matrix(G, T, title='', save_path='trained_similarity_matrix_FINAL.png')
