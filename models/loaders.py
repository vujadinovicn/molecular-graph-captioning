from models.llm import LLMDecoder
from models.graph_encoder import GINEEncoder
from models.adapter import ModalityAdapter
from models.model import MolecularCaptioningModel
from data.utils import get_num_embeddings_list


def get_graph_encoder(config, device):
    graph_hidden_dim = config['train_contrastive'].get('graph_hidden_dim', 256)
    graph_out_dim = config['train_contrastive'].get('graph_out_dim', 512)

    atom_num_embeddings_list, bond_num_embeddings_list = get_num_embeddings_list()
    model = GINEEncoder(
        atom_num_embeddings_list,
        bond_num_embeddings_list,
        hidden_dim=graph_hidden_dim,
        out_dim=graph_out_dim,
        pooling="mean",
    ).to(device)

    return model

def get_modality_adapter(config, device):
    graph_hidden_dim = config['train_contrastive'].get('graph_hidden_dim', 256)
    adapter_hidden_dim = config['train_contrastive'].get('adapter_hidden_dim', 768)
    llama_hidden_dim = config['train_contrastive'].get('llama_hidden_dim', 768)
    
    # here its graph_hidden_dim because thats the size when we obtain per node embs. If we were to use merged embs, it would be graph_out_dim
    model = ModalityAdapter(
        graph_dim=graph_hidden_dim, 
        hidden_dim=adapter_hidden_dim, 
        llm_dim=llama_hidden_dim
    ).to(device)

    return model

def get_molecular_captioning_model(config, device):
    graph_encoder = get_graph_encoder(config, device)
    modality_adapter = get_modality_adapter(config, device)
    # TODO: Kshitij. Model the class in llm.py and load it here accordingly
    llama_decoder = LLMDecoder(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # llama_decoder = LLMDecoder(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    
    model = MolecularCaptioningModel(
        graph_encoder=graph_encoder,
        modality_adapter=modality_adapter,
        llm=llama_decoder,
    ).to(device)

    # TODO: Load
    model.llama_decoder.requires_grad_(False)

    return model