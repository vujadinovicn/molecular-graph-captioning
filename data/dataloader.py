from data.collater import GraphormerMolecularCaptioningCollator, PyGMolecularCaptioningCollator
from torch.utils.data import DataLoader

class MolecularCaptioningDataLoader(DataLoader): 
    def __init__(self, dataset, mode, batch_size, shuffle, follow_batch, exclude_keys, use_graphormer=True, **kwargs):
        kwargs.pop("collate_fn", None)

        if use_graphormer:
            collate_fn = GraphormerMolecularCaptioningCollator(
                tokenizer=dataset.description_tokenizer,
                mode=mode
            )
        else:
            collate_fn = PyGMolecularCaptioningCollator(
                dataset=dataset,
                tokenizer=dataset.description_tokenizer,
                mode=mode,
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
            )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )

    def __len__(self):
        return super().__len__()

    def __iter__(self):
        return super().__iter__()