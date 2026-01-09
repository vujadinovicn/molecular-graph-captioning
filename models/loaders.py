from models.llm import LLMDecoder
from models.graph_encoder import GINEEncoder
from models.adapter import ModalityAdapter
from models.model import MolecularCaptioningModel
from data.utils import get_num_embeddings_list
from transformers import AutoTokenizer

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

def get_llm(config):
    tokenizer = AutoTokenizer.from_pretrained(config['models']['llm_name'])
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.add_tokens(["<|reserved_special_token_1|>"], special_tokens=True)
    
    llm_model = LLMDecoder(model_name=config['models']['llm_name'])
    llm_model.resize_token_embeddings(len(tokenizer))

    return llm_model, tokenizer

def get_molecular_captioning_model(config, device):
    graph_encoder = get_graph_encoder(config, device)
    # modality_adapter = get_modality_adapter(config, device)
    llm_model, tokenizer = get_llm(config)

    model = MolecularCaptioningModel(
        graph_encoder=graph_encoder,
        llm_model=llm_model,
        tokenizer=tokenizer,
        node_dim=256,         
        global_feat_1_dim=20,
        global_feat_2_dim=1024 
    ).to(device)

    return model, tokenizer