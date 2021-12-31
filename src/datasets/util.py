import pickle

import numpy as np
import os
import pickle
import torch

from datasets.buffer import BufferDataset


def get_dataset(data_path, N, save_dir):
    data = load_data_from_file(data_path, N)
    # TODO
    print(f'saving to: ', os.path.join(save_dir, 'train_data.pkl'))
    pickle.dump(data, open(os.path.join(save_dir, 'train_data.pkl'), 'wb'))
    train = BufferDataset(data, split='train')
    val = BufferDataset(data, split='val')
    
    return train, val


def load_data_from_file(file_path, N, shuffle=True):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        data = np.array(data)

    N_available = len(data)
    if N < N_available:
        if shuffle:
            idxs = torch.randperm(len(data))
            data = data[idxs]
        data = data[:N]
    elif N > N_available:
        print(
            f"Warning: requested size of dataset N={N} is greater than {N_available}, the size of the dataset available."
            " Using entire dataset."
        )

    obs = []
    act = []

    for demo in data:
        obs.extend(demo["obs"])
        act.extend(demo["act"])
    data = {"obs": torch.stack(obs), "act": torch.stack(act)}

    if shuffle:
        idxs = torch.randperm(len(data["obs"]))
    else:
        idxs = torch.arange(len(data["obs"]))
    data["obs"] = data["obs"][idxs]
    data["act"] = data["act"][idxs]

    return data
