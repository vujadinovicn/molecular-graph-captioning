import torch.nn as nn
from transformers import GraphormerModel
from peft import get_peft_model

class GraphormerEncoder(nn.Module):
    def __init__(
            self, 
            hf_model_name="clefourrier/graphormer-base-pcqm4mv2", 
            lora_config=None
        ):
        super().__init__()
        self.model = GraphormerModel.from_pretrained(hf_model_name)
       
        if lora_config:
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, batch):
        out = self.model(**batch)
        emb = out.last_hidden_state
        return emb

# lora_config = LoraConfig(
#             r = 32,
#             lora_alpha = 64,
#             target_modules = [
#             "q_proj", "k_proj", "v_proj", "out_proj",   # attention projections
#             "fc1", "fc2"     # MLP projections;
#             ],
#             lora_dropout = 0.05,
#             bias = "none",
#             task_type = TaskType.FEATURE_EXTRACTION
#         )
