import torch
import torch.nn as nn
from models.mc_dcnn import BitNet
from utils import split_sequence
import pandas as pd

# Import dataset

# Loop through dataset and format into subsequences

# Split into train/test

# Build model

# Train model with model.train()

# Test model with model.eval()


net = BitNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
loss_func = nn.MSELoss()
