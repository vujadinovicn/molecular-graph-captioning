import torch.nn as nn
from transformers import GraphormerModel
from peft import LoraConfig, get_peft_model, TaskType

class GraphormerMolecularCaptioningEncoder(nn.Module):
    def __init__(
            self, 
            hf_model_name="clefourrier/graphormer-base-pcqm4mv2", 
            out_dim=768,
            use_lora=True
        ):
        super().__init__()
        self.model = GraphormerModel.from_pretrained(hf_model_name)
        hidden = self.model.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim) if hidden != out_dim else nn.Identity()

        if use_lora:
            lora_config = LoraConfig(
                r = 32,
                lora_alpha = 64,
                target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",   # attention projections
                "fc1", "fc2"     # MLP projections;
                ],
                lora_dropout = 0.05,
                bias = "none",
                task_type = TaskType.FEATURE_EXTRACTION
            )
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, batch):
        out = self.model(**batch)
        emb = out.last_hidden_state[:, 0, :]
        emb = self.proj(emb)
        return emb
