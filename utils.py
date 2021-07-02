import numpy as np
from torch import tensor
import pandas as pd

# Return formatted data
def split_sequence(sequence, n_steps):
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

print(X, y)