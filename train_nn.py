"""
The training function for the network. handles data preprocessing and model training.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.ts_dcnn import BitNet

PATH = "./runs/"
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()

# Import dataset
data = pd.read_csv("data/training.csv")
data = data.drop_duplicates(subset="load_date")

data.set_index("load_date", inplace=True)
# Shift the price by one timestep
data["prior_price"] = data["usd"].shift(periods=1, fill_value=0)

data.dropna(inplace=True)
y = data["usd"]
X = data.drop(["crypto", "id", "AVG(s2.score)", "row_num", "usd"], axis=1)

# Shift the price to be t-1

X = se.fit_transform(X)
X = ss.fit_transform(X)

X = torch.from_numpy(X).float()
# y = torch.tensor(y).float()


def init_weights(mod):
    """Initialize weights using xavier"""
    if isinstance(mod, nn.Linear):
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet()
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-5)
criterion = nn.MSELoss()

# Train model with model.train()
model.train()
for epoch in range(150):
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

    print(
        "epoch: ", epoch, " loss: ", np.sqrt(RUNNING_LOSS / len(X))
    )  # print out loss for each epoch

torch.save(model.state_dict(), PATH + "base.pt")

# Test model
with torch.no_grad():
    pass
