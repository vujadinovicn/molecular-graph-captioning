import torch.nn as nn
from transformers import GraphormerModel

class MolGraphormerEncoder(nn.Module):
    def __init__(self, hf_model_name="clefourrier/graphormer-base-pcqm4mv2", out_dim=768):
        super().__init__()
        self.model = GraphormerModel.from_pretrained(hf_model_name)
        hidden = self.model.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim) if hidden != out_dim else nn.Identity()

    def forward(self, batch):
        out = self.model(**batch)
        emb = out.last_hidden_state[:, 0, :]
        emb = self.proj(emb)
        return emb
