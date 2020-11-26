import numpy as np
import torch
from torch.utils.data import Dataset


def get_d_and_b(interval_low=-2.2, interval_up=2.2, dt=0.01):
    set_d = np.arange(interval_low, interval_up, dt)
    set_b = np.array([interval_low, interval_up])
    return set_d, set_b


def gen_train_data(dt=0.01, num=1):
    data_d, data_b = [], []
    for _ in range(num):
        d, b = get_d_and_b(dt=dt)
        data_d.append(d)
        data_b.append(b)
    return np.expand_dims(np.array(data_d), axis=2), np.expand_dims(np.array(data_b), axis=2)


class PdeDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    d, b = get_d_and_b()
    print(d.shape, b.shape)
