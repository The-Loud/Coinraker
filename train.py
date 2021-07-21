"""
The training function for the network. handles data preprocessing and model training.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.mc_dcnn import BitNet
from utils import split_sequence

PATH = "./runs/"
# Import dataset
data = pd.read_csv("data/base_data.csv")
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()
# Loop through dataset and format into subsequences
STEPS = 24  # 1 day
y = data["usd"]
X = data.drop(["load_date", "crypto", "id"], axis=1)

X = se.fit_transform(X)
X = ss.fit_transform(X)

y = y.to_numpy()
X, y = split_sequence(X, y, STEPS)

X = X.permute(0, 2, 1)

# test = X[0].reshape(1, 1, -1)
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=420
)

# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(X.shape[1])  # .apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Train model with model.train()
model.train()
for epoch in range(250):
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

torch.save(model.state_dict(), PATH + "base.pt")

# Test model
with torch.no_grad():
    pass
