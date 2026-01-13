import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
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
        return node_emb, node_mask, g_proj

class Projector(nn.Module):
    """Projects features from source_dim to llm_dim."""
    def __init__(self, source_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoleculeLlama3(nn.Module):
    def __init__(
        self, 
        graph_encoder, 
        llm_model, 
        tokenizer,
        node_dim=256,       # Output dim of your GINE atom_mlp/hidden_dim
        global_feat_1_dim=20, 
        global_feat_2_dim=1024
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.llm_dim = llm_model.config.hidden_size 
        
        self.graph_encoder = graph_encoder

        self.llm = llm_model
        
        self.node_projector = Projector(node_dim, self.llm_dim)
        self.feat1_projector = Projector(global_feat_1_dim, self.llm_dim)
        self.feat2_projector = Projector(global_feat_2_dim, self.llm_dim)

        self.special_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_1|>")

    def forward(self, batch):
        graph_batch = batch["graph_batch"]
        node_emb, node_mask, _ = self.graph_encoder(graph_batch)

        proj_nodes = self.node_projector(node_emb) 

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            special_token_mask = (input_ids[i] == self.special_token_id)
            num_special_tokens = special_token_mask.sum().item()

            valid_nodes = proj_nodes[i][node_mask[i]] 
            
            if num_special_tokens == valid_nodes.shape[0]:
                inputs_embeds[i][special_token_mask] = valid_nodes.to(inputs_embeds.dtype)
            else:
                print("kraj")
                quit()

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
