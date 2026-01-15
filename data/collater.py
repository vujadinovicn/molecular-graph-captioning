from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

class MolecularCaptioningCollator:
    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, batch_list):
        text_inputs = []
        labels_list = []
        descriptions = []
        
        for data in batch_list:
            text_inputs.append({
                "input_ids": data.input_ids,
                "attention_mask": data.attention_mask
            })
            labels_list.append(data.labels)

            if self.mode != "test":
                description = data.description_input_ids
                if description.dim() == 2:
                    description = description.squeeze(0)
                descriptions.append(description)

        graph_batch = Batch.from_data_list(
            batch_list,
            exclude_keys=["description_input_ids"]
        )

        text_batch = self.tokenizer.pad(
            text_inputs,
            padding=True,
            return_tensors="pt"
        )

        if self.tokenizer.padding_side == "right":
            labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        else:
            reversed_labels = [lbl.flip(0) for lbl in labels_list]
            padded_reversed = pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
            labels_padded = padded_reversed.flip(1)

        batch = {
            "graph_batch": graph_batch,
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "labels": labels_padded
        }

        # include description only for train and val splits
        if self.mode == "test":
            return batch
        
        batch['description_input_ids'] = pad_sequence(descriptions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch['description_attention_mask'] = (batch['description_input_ids'] != self.tokenizer.pad_token_id).long()
        return batch
    