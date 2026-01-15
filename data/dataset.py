import pickle
from torch.utils.data import Dataset
import torch.utils.data
import os

class MolecularCaptioningDataset(Dataset):
    def __init__(
            self, 
            graphs_path, 
            split="train",
            tokenizer=None, 
            max_length=1024,
            max_description_length=512,
            captions_path="/home/shishirk/adityasr/kshitij_molecular_captioning/all_descriptions.txt" # text file containing correct captions
    ):
        self.graphs_path = graphs_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_description_length = max_description_length
        self.split = split
        
        # incontext captions
        self.all_captions = []
        if os.path.exists(captions_path):
            with open(captions_path, 'r', encoding='utf-8') as f:
                self.all_captions = [line.strip() for line in f if line.strip()]
        else:
            print(f"Warning: Captions file not found at {captions_path}. Using fallback.")
            self.all_captions = ["The molecule is a generic chemical compound."]

        # Define the static part of the system instruction
        self.fixed_system_message = (
            "You are a helpful assistant that captions molecules based on their structure. "
            "Provide concise and informative descriptions. Following are a few examples of captions:"
            "## Example 1: The molecule is a 4-O-[(E)-2-methyl-2-butenoyl]ascaroside derived from (2E,8R)-8-hydroxynon-2-enoic acid. It is a metabolite of the nematode Caenorhabditis elegans. It has a role as a Caenorhabditis elegans metabolite. It is a 4-O-[(E)-2-methyl-2-butenoyl]ascaroside and an alpha,beta-unsaturated monocarboxylic acid. It derives from an ascr#3 and a (2E,8R)-8-hydroxynon-2-enoic acid."
            "## Example 2: The molecule is an alkanesulfonic acid in which the alkyl group directly linked to the sulfo functionality is methyl. It has a role as an Escherichia coli metabolite. It is an alkanesulfonic acid and a one-carbon compound. It is a conjugate acid of a methanesulfonate."
        )

        self.load_graphs()

    def load_graphs(self):
        with open(os.path.join(self.graphs_path, self.split + "_graphs.pkl"), 'rb') as f:
            self.graphs = pickle.load(f)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        graph.num_patch_tokens = graph.num_nodes # +2 for global features
        
        caption = getattr(graph, "description", "")
        if not caption:
            caption = "No description available."
            
        placeholder_token = '<|reserved_special_token_1|>'
        num_placeholders = graph.num_nodes
        user_message_content = f"Caption the following molecule: {placeholder_token * num_placeholders}"

        full_conversation = [
            {"role": "system", "content": self.fixed_system_message},
            {"role": "user", "content": user_message_content},
            {"role": "assistant", "content": caption}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        prompt_conversation = full_conversation[:2]

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_conversation,
            tokenize=True,
            add_generation_prompt=True, 
            return_tensors="pt"
        ).squeeze(0)

        prompt_len = prompt_ids.shape[0]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        graph.input_ids = input_ids
        graph.labels = labels
        graph.attention_mask = torch.ones_like(input_ids)

        if getattr(graph, "description", None) is not None:
            description_ids = self.tokenizer(
                caption + self.tokenizer.eos_token,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            
            if description_ids.size(-1) > self.max_description_length:
                description_ids = description_ids[:, :self.max_length]

            graph.description_input_ids = description_ids
        

        return graph