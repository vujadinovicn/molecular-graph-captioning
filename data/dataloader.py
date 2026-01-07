from torch.utils.data import DataLoader
from data.collater import PyGMolecularCaptioningCollator

class MolecularCaptioningDataLoader(DataLoader): 
    def __init__(self, dataset, mode, batch_size, shuffle, follow_batch=[], exclude_keys=[], **kwargs):
        kwargs.pop("collate_fn", None)

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
    
def get_dataloader(dataset, mode, batch_size):
    return MolecularCaptioningDataLoader(
        dataset=dataset,
        mode=mode,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )