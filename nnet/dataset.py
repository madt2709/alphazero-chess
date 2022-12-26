from torch.utils.data import Dataset
import numpy as np


class BoardData(Dataset):
    def __init__(self, dataset):
        # dataset is a list of lists of form [s, p, v].
        super().__init__()
        self.X = []
        self.Y_p, self.Y_v = [], []
        for data in dataset:
            s, p, v = data
            self.X.append(s)
            self.Y_p.append(p)
            self.Y_v.append(v)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_p[idx], self.Y_v[idx]

    def __len__(self):
        return len(self.X)
