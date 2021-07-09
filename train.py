import torch
import torch.nn as nn
from models.mc_dcnn import BitNet
from utils import split_sequence
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Import dataset
data = pd.read_csv("data/Bitcoin_dataset_updated 2.csv")
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()
# Loop through dataset and format into subsequences
steps = 24  # 1 day
y = data["BTC price"]
X = data.drop(["Date", "BTC price"], axis=1)

X = se.fit_transform(X)
X = ss.fit_transform(X)

X, y = X, y.to_numpy()
X, y = split_sequence(X, y, steps)

X = X.permute(0, 2, 1)

# test = X[0].reshape(1, 1, -1)
# Split into train/test


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m)
        m.bias.data.fill_(0.01)


# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(X.shape[1])  # .apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train model with model.train()
for epoch in range(15):  # training the model for 15 times with the same data
    running_loss = 0.0
    for i, value in enumerate(
        X
    ):  # looping through all the mini-batches of the entire batch of data
        inputs = value.unsqueeze(0)  # getting input value
        labels = y[i]  # getting label value
        prediction = model(inputs)  # passing inputs to our model to get prediction
        loss = criterion(
            prediction.squeeze(), labels
        )  # loss is calculating using MSELoss function
        running_loss += loss.item() * inputs.size(
            0
        )  # adding loss value to print it later
        optimizer.zero_grad()  # reset all gradient calculation
        loss.backward()  # this is backpropagation to calculate gradients
        optimizer.step()  # applying gradient descent to update weights and bias values

    abs_deltas = prediction - labels
    print(prediction)
    print(labels)
    print(abs_deltas)

    print(
        "epoch: ", epoch, " loss: ", running_loss / len(X)
    )  # print out loss for each epoch

# out = model(X)
# print(out)


# Test model with model.eval()
