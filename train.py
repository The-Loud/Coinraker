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
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()

# Import dataset
data = pd.read_csv("data/training.csv")
data = data.drop_duplicates(subset="load_date")

data.set_index("load_date", inplace=True)
# Shift the price by one timestep
data["prior_price"] = data["usd"].shift(periods=1, fill_value=0)

# Loop through dataset and format into subsequences
STEPS = 24  # 1 day

data.dropna(inplace=True)
y = data["usd"]
X = data.drop(["crypto", "id", "AVG(s2.score)", "row_num", "usd"], axis=1)

# Shift the price to be t-1

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


def init_weights(mod):
    """Initialize weights using xavier"""
    if isinstance(mod, nn.Linear):
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(X.shape[1])  # .apply(init_weights)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=1e-5)
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X.to(device)
y.to(device)
model.to(device)

preds = []
# Train model with model.train()
model.train()
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
    preds.append(prediction.detach().numpy())

    print(
        "epoch: ", epoch, " loss: ", np.sqrt(RUNNING_LOSS / len(X))
    )  # print out loss for each epoch

torch.save(model.state_dict(), PATH + "base_729.pt")

# plt.plot(X[:, 1, 1], preds, color='red')
# plt.plot(X[:, 1, 1], y.detach().numpy())
# plt.show()

# Test model
with torch.no_grad():
    pass
