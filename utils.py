import numpy as np
import torch
from torch import tensor
import pandas as pd


def split_sequence(x_seq: np.array, y_seq: np.array, n_steps: int) -> torch.tensor:
    """
    Splits up a time series numpy array into tensors of subsequences.
    :param x_seq: feature sequence
    :param y_seq: target sequence
    :param n_steps: the number of time steps in each subsequence
    :return: two tensors
    """
    X, y = list(), list()
    for i in range(len(x_seq)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(x_seq) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = x_seq[i:end_ix], y_seq[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float()


# 1 day
steps = 24

data = pd.read_csv("data/Bitcoin_dataset_updated 2.csv")
y = data["BTC price"]
X = data.drop(["Date", "BTC price"], axis=1)

X, y = X.to_numpy(), y.to_numpy()
X, y = split_sequence(X, y, steps)

# Change the shape to be samples x channels x subsequence
X = X.permute(0, 2, 1)

# print(X, y)
