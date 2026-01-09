import torch
import torch.nn as nn
import torch.nn.functional as F
from models.readout import readout_embeddings
from adapter import Projector

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

        # lora_part   
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     target_modules=["q_proj", "v_proj"] # Target attention layers
        # )
        # self.llm = get_peft_model(llm_model, peft_config)

        self.node_projector = Projector(node_dim, self.llm_dim)
        self.feat1_projector = Projector(global_feat_1_dim, self.llm_dim)
        self.feat2_projector = Projector(global_feat_2_dim, self.llm_dim)
        self.use_global_feats = use_global_feats

        self.special_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_1|>")


    def forward_instruct(self, batch):
        graph_batch = batch["graph_batch"]

        node_emb, node_mask, _ = self.graph_encoder(graph_batch)
        proj_nodes = self.node_projector(node_emb) 

        g1 = graph_batch.global_feat_1
        g2 = graph_batch.global_feat_2
        if self.use_global_feats:
            proj_g1 = self.feat1_projector(g1).unsqueeze(1)
            proj_g2 = self.feat2_projector(g2).unsqueeze(1)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            special_token_mask = (input_ids[i] == self.special_token_id)
            num_special_tokens = special_token_mask.sum().item()
            
            valid_nodes = proj_nodes[i][node_mask[i]] 
            
            if self.use_global_feats:
                fused_graph_seq = torch.cat([proj_g1[i], proj_g2[i], valid_nodes], dim=0)
            else:
                fused_graph_seq = valid_nodes
                
            if num_special_tokens == fused_graph_seq.shape[0]:
                inputs_embeds[i][special_token_mask] = fused_graph_seq.to(inputs_embeds.dtype)
            else:
                min_len = min(num_special_tokens, fused_graph_seq.shape[0])
                
                special_token_indices = torch.nonzero(special_token_mask).squeeze()
                indices_to_replace = special_token_indices[:min_len]
                features_to_use = fused_graph_seq[:min_len]
                
                inputs_embeds[i][indices_to_replace] = features_to_use.to(inputs_embeds.dtype)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def get_graph_embeddings(self, graph_batch, readout_fn, normalize=True):
        graph_batch = graph_batch["graph_batch"]

        node_emb, node_mask, _ = self.graph_encoder(graph_batch)
        proj_tokens = self.node_projector(node_emb)

        if self.use_global_feats:
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
            embeddings = readout_embeddings(proj_tokens, node_mask, "mix") 

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
            output_layer=text_layer,
            readout_fn=readout_fn,
            normalize=True
        )

        return graph_emb, text_emb

    def forward_instruct(**kwargs):
        # TODO
        pass

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True