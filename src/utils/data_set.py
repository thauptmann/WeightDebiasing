import torch
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(self, X):
        self.x = torch.Tensor(X)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
