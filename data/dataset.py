import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from transformers.models.graphormer.collating_graphormer import preprocess_item
import os

class MolecularCaptioningDataset(Dataset):
    def __init__(
            self, 
            graphs_path, 
            split="train",
            description_tokenizer=None, 
            max_description_length=512, 
            use_graphormer=True,
            **kwargs
    ):
        self.graphs_path = graphs_path
        self.description_tokenizer = description_tokenizer
        self.max_description_length = max_description_length
        self.split = split
        self.use_graphormer = use_graphormer

        self.load_graphs()

    def load_graphs(self):
        with open(os.path.join(self.graphs_path, self.split+"_graphs.pkl"), 'rb') as f:
            self.graphs = pickle.load(f)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        chat = self.build_and_tokenize_chat_prompt(graph)
        description = self.tokenize_description(graph)

        if self.use_graphormer:
            graph_item = self.pyg_to_graphormer_item(graph)
            graph_item['id'] = graph.id
            graph_item.update(chat)
            graph_item.update(description)
            return graph_item
        else:
            graph.id = graph.id
            graph.prompt_input_ids = chat["prompt_input_ids"]
            graph.description_input_ids = description["description_input_ids"]
            return graph

    def build_and_tokenize_chat_prompt(self, graph):
        # TODO: Write the system message and change user message's placeholder_token
        system_message = "Captio the molecule."
        placeholder_token: str = '<|reserved_special_token_1|>'
        num_nodes = graph.num_nodes # we can change this
        user_message = ("Molecule graph embeddings: " + placeholder_token * (num_nodes + 2))
        
        prompt =  [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        prompt_input_ids = self.description_tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            return_tensors="pt"
        )      

        return {"prompt_input_ids": prompt_input_ids}
     
    def tokenize_description(self, graph):
        description = getattr(graph, "description", None)

        if not description:
            return {"description_input_ids": None}

        input_ids = self.description_tokenizer(
            [description],
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        if input_ids.size(-1) > self.max_description_length:
            input_ids = input_ids[:, :self.max_description_length]
            description = self.description_tokenizer.decode(input_ids[0], skip_special_tokens=True)

        description_ids = self.description_tokenizer(
            [description + self.description_tokenizer.eos_token],
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]

        return {"description_input_ids": description_ids}

    def pyg_to_graphormer_item(self, graph):
        """
        Convert a PyTorch Geometric Data object to Graphormer input format
        and run Graphormer preprocessing.
        """
        node_feat = graph.x.to(torch.long)
        edge_feat = graph.edge_attr.to(torch.long)
        edge_index = graph.edge_index.to(torch.long)

        item = {
            "edge_index": edge_index,
            "node_feat": node_feat,
            "edge_feat": edge_feat,
            "num_nodes": graph.num_nodes,
            "y": torch.tensor(0)
        }

        item = preprocess_item(item)
        return item