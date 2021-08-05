"""
This module contains a simple function to format the data into a ready-made tensor.
The sliding window can be defined and the returned X will be a tensor
of (subsequence slices x channels x subsequence length)
"""
import numpy as np
import pandas as pd
import torch


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
