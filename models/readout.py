import torch

def readout_embeddings(embeddings, attention_mask, readout_fn):
    if readout_fn == "last":
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        return embeddings[batch_indices, last_token_indices, :]

    elif readout_fn == "mean":
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)
        return sum_embeddings / count_attn_mask
    
    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="mean")
        diff_embeddings = embeddings - mean_embeddings.unsqueeze(1)
        diff_embeddings_2 = diff_embeddings.pow(2) 
        masked_diff_embeddings_2 = diff_embeddings_2 * attention_mask.unsqueeze(-1)
        sum_diff_embeddings_2 = masked_diff_embeddings_2.sum(dim=1) 
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)
        return (sum_diff_embeddings_2 / count_attn_mask).sqrt()

    elif readout_fn == "mix": 
        mean_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="mean")
        std_embeddings = readout_embeddings(embeddings=embeddings, attention_mask=attention_mask, readout_fn="std")
        return torch.cat([mean_embeddings, std_embeddings], dim=1)
