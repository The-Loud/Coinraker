"""
The training function for the network. handles data preprocessing and model training.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.mc_dcnn import BitNet
from utils import split_sequence

PATH = "./runs/"
# Import dataset
data = pd.read_csv("data/Bitcoin_dataset_updated 2.csv")
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()
# Loop through dataset and format into subsequences
STEPS = 24  # 1 day
y = data["BTC price"]
X = data.drop(["Date", "BTC price"], axis=1)

X = se.fit_transform(X)
X = ss.fit_transform(X)

y = y.to_numpy()
X, y = split_sequence(X, y, STEPS)

X = X.permute(0, 2, 1)

# test = X[0].reshape(1, 1, -1)
# Split into train/test

# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(X.shape[1])  # .apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Train model with model.train()
for epoch in range(15):
    RUNNING_LOSS = 0.0
    for i, value in enumerate(X):
        inputs = value.unsqueeze(0)
        labels = y[i]
        prediction = model(inputs)
        loss = criterion(prediction.squeeze(), labels)
        RUNNING_LOSS += loss.item() * inputs.size(0)
        optimizer.zero_grad()
        loss.backward()  # this is backpropagation to calculate gradients
        optimizer.step()  # applying gradient descent to update weights and bias values

    abs_deltas = prediction - labels
    print(prediction)
    print(labels)
    print(abs_deltas)

    print(
        "epoch: ", epoch, " loss: ", RUNNING_LOSS / len(X)
    )  # print out loss for each epoch

torch.save(model.state_dict(), PATH)

# Test model with model.eval()
