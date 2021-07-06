import torch
import torch.nn as nn
from models.mc_dcnn import BitNet
from utils import split_sequence
import pandas as pd

# Import dataset
data = pd.read_csv('data/Bitcoin_dataset_updated 2.csv')


# Loop through dataset and format into subsequences
steps = 24  # 1 day
y = data['BTC price']
X = data['n-transactions']

X, y = X.to_numpy(), y.to_numpy()
X, y = split_sequence(X, y, steps)

test = X[0].reshape(1, 1, -1)
# Split into train/test

# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(test.shape[1])

# Train model with model.train()

out = model(test)
print(out.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
loss_func = nn.MSELoss()

# Test model with model.eval()
