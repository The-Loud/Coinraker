import numpy as np
import torch
from torch import tensor
import pandas as pd


def split_sequence(sequence: np.array, n_steps: int) -> torch.tensor:
    """
    Splits up a time series numpy array into tensors of subsequences.
    :param sequence: the sequence to be split
    :param n_steps: the number of time steps in each subsequence
    :return: two tensors
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# 1 day
steps = 24

data = pd.read_csv('data/Bitcoin_dataset_updated 2.csv')
data = data['BTC price']

data = data.to_numpy()

X, y = split_sequence(data, steps)

X, y = torch.from_numpy(X), torch.from_numpy(y)

print(X, y)