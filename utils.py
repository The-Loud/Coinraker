"""
This module contains a simple function to format the data into a ready-made tensor.
The sliding window can be defined and the returned X will be a tensor
of (subsequence slices x channels x subsequence length)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_sequence(x_seq: np.array, y_seq: np.array, n_steps: int) -> torch.tensor:
    """
    Splits up a time series numpy array into tensors of subsequences.
    :param x_seq: feature sequence
    :param y_seq: target sequence
    :param n_steps: the number of time steps in each subsequence
    :return: two tensors
    """
    x_train, y_train = list(), list()
    for i in range(len(x_seq)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(x_seq) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = x_seq[i:end_ix], y_seq[end_ix]
        x_train.append(seq_x)
        y_train.append(seq_y)
    return (
        torch.from_numpy(np.array(x_train)).float(),
        torch.from_numpy(np.array(y_train)).float(),
    )


class CoinTrainset(Dataset):
    """formats the data into proper tensor shape and returns a sample of (1 x 5 x 24)"""

    def __init__(self, training_data: np.ndarray, y_train: np.array) -> None:
        super().__init__()

        self.training_data = training_data
        self.y_train = y_train
        self.steps = 24

        self.x_train, self.y_train = split_sequence(
            self.training_data, self.y_train.to_numpy(), self.steps
        )

        self.x_train = self.x_train.permute(0, 2, 1)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


class CoinTestset(Dataset):
    """formats the data into proper tensor shape and returns a sample of (1 x 5 x 24)"""

    def __init__(self, test_data: np.ndarray, y_test: np.array) -> None:
        super().__init__()

        self.test_data = test_data
        self.y_test = y_test
        self.steps = 24

        self.x_test, self.y_test = split_sequence(
            self.test_data, self.y_test.to_numpy(), self.steps
        )

        self.x_test = self.x_test.permute(0, 2, 1)

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]
