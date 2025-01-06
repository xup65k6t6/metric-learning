import torch


class StructuredDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        """
        Create a dataset suitable for Lifted Structure Loss

        Args:
            original_dataset (Dataset): The original dataset
        """
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, aug_img, label = self.original_dataset[idx]
        return aug_img, label
