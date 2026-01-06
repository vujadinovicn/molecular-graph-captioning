import torch
from peft import LoraConfig, TaskType
from graph_encoder import GraphormerEncoder
from llm import LLMDecoder
from adapter import ModalityAdapter
from model import MolecularCaptioningModel

def load_model(config):
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
    graph_encoder = GraphormerEncoder(lora_config=None) # TODO: change
    llama_decoder = LLMDecoder(model_name="/")
    modality_adapter = ModalityAdapter(graph_dim=, intermediate_dim=, llm_dim=)
    
    model = MolecularCaptioningModel(
        graph_encoder=graph_encoder,
        modality_adapter=modality_adapter,
        llm=llama_decoder,
    )

    if args["load_model_checkpoint_path"]:
        print(f"Loading {args['load_model_checkpoint_path']}")
        model_state_dict = torch.load(
            args["load_model_checkpoint_path"], 
            weights_only=True, 
            map_location="cpu"  # load to CPU first
            # will be loaded to where the weights were saved from if not specified
        )
        model.load_state_dict(model_state_dict)

    # WARNING: esm and llama weights are fixed
    model.esm_encoder.requires_grad_(False)
    model.llama_decoder.requires_grad_(False)

    return model