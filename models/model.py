import torch
import torch.nn as nn
import torch.nn.functional as F
from models.readout import readout_embeddings
from models.projector import Projector

class MolecularCaptioningModel(nn.Module):
    def __init__(
        self,
        graph_encoder,
        llm_model,
        tokenizer,
        node_dim=256,
        global_feat_1_dim=20,
        global_feat_2_dim=1024,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.llm_dim = llm_model.config.hidden_size 
        self.llm = llm_model
        
        self.graph_encoder = graph_encoder
        self.node_projector = Projector(node_dim, self.llm_dim)
        
        # self.feat1_projector = Projector(global_feat_1_dim, self.llm_dim)
        # self.feat2_projector = Projector(global_feat_2_dim, self.llm_dim)

        self.special_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_1|>")

    def get_graph_embeddings(self, graph_batch, readout_fn, normalize=True):
        node_emb, node_mask, _ = self.graph_encoder(graph_batch)
        proj_tokens = self.node_projector(node_emb)

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
            graph_batch=batch["graph_batch"],
            readout_fn=readout_fn,
            normalize=True
        )

        description_emb = self.get_description_embeddings(
            input_ids=batch["description_input_ids"],
            attention_mask=batch["description_attention_mask"],
            layer=text_layer,
            readout_fn=readout_fn,
            normalize=True
        )

        return graph_emb, description_emb

    def forward_instruct(self, batch):
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
                # TODO: 
                pass

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs