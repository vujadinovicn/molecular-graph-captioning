import torch
import torch.nn as nn
import torch.nn.functional as F
from models.readout import readout_embeddings

class MolecularCaptioningModel(nn.Module):
    def __init__(self, graph_encoder, modality_adapter, llm):
            super().__init__()
            self.graph_encoder = graph_encoder
            self.modality_adapter = modality_adapter
            self.llm = llm

    def get_graph_embeddings(self, graph_batch, readout_fn, normalize=True):
        graph_tokens = self.graph_encoder(graph_batch)
        proj_tokens = self.modality_adapter(graph_tokens)

        embeddings = readout_embeddings(proj_tokens, graph_batch["node_mask"], readout_fn)

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
            graph_batch, 
            description_input_ids, 
            description_attention_mask, 
            graph_mask, 
            text_layer, 
            readout_fn
        ):
        graph_emb = self.get_graph_embeddings(
            graph_batch=graph_batch,
            graph_mask=graph_mask,
            readout_fn=readout_fn,
            normalize=True
        )

        text_emb = self.get_description_embeddings(
            input_ids=description_input_ids,
            attention_mask=description_attention_mask,
            output_layer=text_layer,
            readout_fn=readout_fn,
            normalize=True
        )

        return {"graph_emb": graph_emb, "text_emb": text_emb}

    def forward_instruct(**kwargs):
        # TODO
        pass