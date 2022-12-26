from torch.utils.data import Dataset
import numpy as np


class BoardData(Dataset):
    def __init__(self, dataset):
        # dataset is a list of lists of form [s, p, v].
        super().__init__()
        dataset = np.array(dataset)
        self.X = dataset[:, 0]
        self.Y_p, self.Y_v = dataset[:, 1], dataset[:, 2]

    def __getitem__(self, idx):
        return self.X[idx], self.Y_p[idx], self.Y_v[idx]

    def __len__(self):
        return len(self.X)
