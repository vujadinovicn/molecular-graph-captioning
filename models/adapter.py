import torch.nn as nn

class ModalityAdapter(nn.Module):
    def __init__(self, graph_dim, hidden_dim, llm_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)

        # out = F.normalize(out, p=2, dim=-1)
        return out
