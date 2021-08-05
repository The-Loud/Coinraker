"""
This module runs a basic linear regression on log-transformed and normalized data.
It serves as a baseline to see if the accuracy of the MC DCNN is better.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.ts_dcnn import BitNet

# Load up the data
data = pd.read_csv("./data/training.csv")
data = data.drop_duplicates(subset="load_date")

data.set_index("load_date", inplace=True)
# Shift the price by one timestep
data["prior_price"] = data["usd"].shift(periods=1, fill_value=0)

# Drop the first row which has a prior price of 0
data = data.iloc[1:, :]
X = data.drop(["usd", "crypto", "AVG(s2.score)", "row_num", "id"], axis=1)
y = np.log(data["usd"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=420
)

# Preprocess
si = SimpleImputer(missing_values=np.nan, strategy="mean")
si.fit(X_train)

X_train = si.transform(X_train)
X_test = si.transform(X_test)

# Scale the data
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

model = BitNet()
# Do the thing
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
criterion = nn.MSELoss()

preds = []
trloss = []
vloss = []

model.train()
for epoch in range(10):
    y_true_train = list()
    y_pred_train = list()
    TOTAL_LOSS_TRAIN = 0

    for i, value in enumerate(X_train):
        inputs = value
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
            inputs = value
            labels = y_test[i]
            prediction = model(inputs)
            loss = criterion(prediction.squeeze(), labels)

            y_true_val += list(labels.unsqueeze(0))
            y_pred_val += list(prediction.cpu().data.numpy())
            TOTAL_LOSS_VAL += loss.item()
            if epoch == 9:
                preds.append(prediction.detach().numpy())
        valacc = mean_squared_error(y_true_val, y_pred_val, squared=False)
        vloss.append(valacc)
        valloss = TOTAL_LOSS_VAL / len(X_test)
        print(
            f"Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f} | val_loss: {valloss:.4f} val_rmse: {valacc:.4f}"
        )

plt.plot(trloss)
plt.plot(vloss)
plt.legend(["training", "test"], loc="lower right")
# plt.ylim(0, 5000)
plt.show()

plt.plot(y_test)
plt.plot(preds)
plt.show()
