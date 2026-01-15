import torch
from torch.functional import F
import torch.nn as nn

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