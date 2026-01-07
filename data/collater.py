import torch
import torch.utils.data
import torch_geometric
import torch_geometric.data
import torch_geometric.loader.dataloader

class PyGMolecularCaptioningCollator(torch_geometric.loader.dataloader.Collater):  
    def __init__(self, dataset, tokenizer, mode="train", **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        data_batch = torch_geometric.data.Batch.from_data_list(
            batch, 
            exclude_keys=[
                "prompt_input_ids", 
                "description_input_ids", 
            ]
        )

        text_batch = build_text_batch(batch, self.text_pad_token_id, self.mode)
        data_batch.update(text_batch)

        if self.exclude_keys: 
            data_batch = {
                k: v for k, v in data_batch.items() 
                if k not in self.exclude_keys
            }

        return data_batch
    

def build_text_batch(batch, text_pad_token_id, mode):
    prompt_input_ids = [data["prompt_input_ids"][0] for data in batch]
    pad_prompt_input_ids = pad_sequence(
        prompt_input_ids, 
        padding_value=text_pad_token_id, 
        padding_side="left"
    )
    pad_prompt_attention_mask = pad_sequence(
        [torch.ones_like(data["prompt_input_ids"][0]) for data in batch], 
        padding_value=0, 
        padding_side="left"
    )

    # TODO: Kshitij
    if mode == "test":
        return {
            "input_ids": pad_prompt_input_ids, 
            "attention_mask": pad_prompt_attention_mask, 
        } 

    description_input_ids = [data["description_input_ids"][0] for data in batch]
    pad_description_input_ids = pad_sequence(
        description_input_ids, 
        padding_value=text_pad_token_id, 
        padding_side="right"
    )
    
    pad_description_attention_mask = pad_sequence(
        [torch.ones_like(data["description_input_ids"][0]) for data in batch], 
        padding_value=0, 
        padding_side="right"
    )
    pad_labels = pad_sequence(
        description_input_ids, 
        padding_value=-100,
        padding_side="right"
    )
    
    return {
        "input_ids": torch.cat([pad_prompt_input_ids, pad_description_input_ids], dim=1), 
        "attention_mask": torch.cat([pad_prompt_attention_mask, pad_description_attention_mask], dim=1), 
        "labels": torch.cat([torch.full_like(pad_prompt_input_ids, fill_value=-100), pad_labels], dim=1),
        "description_input_ids": pad_description_input_ids,
        "description_attention_mask": pad_description_attention_mask,
    }
    
def pad_sequence(sequences, padding_value, padding_side="right"):
    max_len = max(sequence.shape[-1] for sequence in sequences)
    padded_sequences = []
    for sequence in sequences: 
        padding = torch.full(
            size=(max_len - sequence.shape[-1],),
            fill_value=padding_value,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        if padding_side == "left":
            padded_sequences.append(torch.cat([padding, sequence], dim=-1))
        elif padding_side == "right":
            padded_sequences.append(torch.cat([sequence, padding], dim=-1))
    return torch.stack(padded_sequences, dim=0)


