"""
The training function for the network. handles data preprocessing and model training.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.mc_dcnn_v2 import BitNet
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

# Drop the first row which has a prior price of 0
data = data.iloc[1:, :]

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


def init_weights(mod):
    """Initialize weights using xavier"""
    if isinstance(mod, nn.Linear):
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


X_train = X[:518]
y_train = y[:518]

X_test = X[518:]
y_test = y[518:]

# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(series=X.shape[1])
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X.to(device)
y.to(device)
model.to(device)

preds = []
trloss = []
vloss = []

model.train()
for epoch in range(30):
    y_true_train = list()
    y_pred_train = list()
    TOTAL_LOSS_TRAIN = 0

    for i, value in enumerate(X_train):
        inputs = value.unsqueeze(0)
        labels = y_train[i]
        prediction = model(inputs)
        loss = criterion(prediction.squeeze(), labels)
        y_true_train += list(labels.unsqueeze(0))
        y_pred_train += list(prediction.cpu().data.numpy())

        # RUNNING_LOSS += loss.item() * inputs.size(0)
        TOTAL_LOSS_TRAIN += loss.item()
        optimizer.zero_grad()
        loss.backward()  # this is backpropagation to calculate gradients
        optimizer.step()  # applying gradient descent to update weights and bias values

        train_acc = mean_squared_error(y_true_train, y_pred_train, squared=False)
        train_loss = TOTAL_LOSS_TRAIN / len(X_train)
    trloss.append(train_acc)

    # Test model
    with torch.no_grad():
        y_true_val = list()
        y_pred_val = list()
        TOTAL_LOSS_VAL = 0

        for i, value in enumerate(X_test):
            inputs = value.unsqueeze(0)
            labels = y_test[i]
            prediction = model(inputs)
            loss = criterion(prediction.squeeze(), labels)

            y_true_val += list(labels.unsqueeze(0))
            y_pred_val += list(prediction.cpu().data.numpy())
            TOTAL_LOSS_VAL += loss.item()
        valacc = mean_squared_error(y_true_val, y_pred_val, squared=False)
        vloss.append(valacc)
        valloss = TOTAL_LOSS_VAL / len(X_test)
        print(
            f"Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f} | val_loss: {valloss:.4f} val_rmse: {valacc:.4f}"
        )


torch.save(model.state_dict(), PATH + "mc2_804.pt")
plt.plot(trloss)
plt.plot(vloss)
plt.legend(["training", "test"], loc="lower right")
plt.ylim(0, 5000)
plt.show()
