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

        self.atom_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr)

        x = self.atom_mlp(x) # to mix features

        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, data.edge_index, e)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += x_res  # residual connection

        if self.pooling == "mean":
            g = global_mean_pool(x, data.batch)
        else:
            g = global_add_pool(x, data.batch)

        g_proj = self.proj(g)
        if self.normalize:
            g_proj = F.normalize(g_proj, dim=-1)

        node_emb, node_mask = to_dense_batch(x, batch=data.batch)
        return node_emb, node_mask, g_proj # return per node embeddings, mask and mean/added embeddings
