import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# TODO: Kshitij. Edit
class LLMDecoder(nn.Module):
    def __init__(self, model_name, device_map="auto"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)
