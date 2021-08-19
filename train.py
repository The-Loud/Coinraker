"""
The training function for the network. handles data preprocessing and model training.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mc_dcnn_v2 import BitNet
from utils import CoinTestset
from utils import CoinTrainset

PATH = "./runs/"
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()

# Import dataset
data = pd.read_csv("data/training.csv")
data = data.dropna()
data = data.drop_duplicates(subset="load_date")
data.set_index("load_date", inplace=True)

# Shift the price by one timestep
data["prior_price"] = data["usd"].shift(periods=1, fill_value=0)

# Drop the first row which has a prior price of 0
# data = data.iloc[:50000, :]
data = data.iloc[1:, :]

# Loop through dataset and format into subsequences
STEPS = 24  # 1 day

y = data["usd"].diff(periods=1)
# y = data['usd']
X = data.drop(["usd", "id", "crypto", "AVG(s2.score)", "row_num"], axis=1)

# Split the datasets
X_train = X[:-200]
y_train = y[:-200]
X_test = X[-200:]
y_test = y[-200:]

# Transform
X_train = se.fit_transform(X_train)
X_train = ss.fit_transform(X_train)
X_test = se.transform(X_test)
X_test = ss.transform(X_test)

train_set = CoinTrainset(X_train, y_train)
test_set = CoinTestset(X_test, y_test)

# We are not shuffling because order matters with time series.
trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
testloader = DataLoader(test_set, batch_size=32, shuffle=True)

'''def init_weights(mod):
    """Initialize weights using xavier"""
    if isinstance(mod, nn.Linear):
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)'''

# Build model. Pass in the number of channels to build the proper Conv layers.
model = BitNet(series=X.shape[1])
# model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preds = []
trloss = []
vloss = []

# data, target = next(iter(trainloader))

for epoch in tqdm(range(20)):
    y_true_train = list()
    y_pred_train = list()
    TOTAL_LOSS_TRAIN = 0

    model.train()
    for i, (data, target) in tqdm(enumerate(trainloader)):
        inputs = data.to(device)
        t_labels = target.to(device)
        t_prediction = model(inputs).to(device)
        loss = criterion(t_prediction, t_labels)
        y_true_train += list(t_labels)
        y_pred_train += list(t_prediction.cpu().data.numpy())

        # RUNNING_LOSS += loss.item() * inputs.size(0)
        TOTAL_LOSS_TRAIN += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = 0  # mean_squared_error(y_true_train, y_pred_train, squared=False)
    train_loss = TOTAL_LOSS_TRAIN / len(X_train)
    trloss.append(train_acc)
    print(t_prediction[-4:])
    print(t_labels[-4:])
    trloss.append(loss.detach().numpy())

    # Test model
    model.eval()
    with torch.no_grad():
        y_true_val = list()
        y_pred_val = list()
        TOTAL_LOSS_VAL = 0

        for i, (data, target) in enumerate(testloader):
            inputs = data.to(device)
            labels = target.to(device)
            prediction = model(inputs).to(device)
            loss = criterion(prediction, labels)

            y_true_val += list(labels)
            y_pred_val += list(prediction.cpu().data.numpy())
            TOTAL_LOSS_VAL += loss.item()
            valacc = 0  # mean_squared_error(y_true_val, y_pred_val, squared=False)
        vloss.append(valacc)
        valloss = TOTAL_LOSS_VAL / len(X_test)
        print(prediction[-4:])
        print(labels[-4:])
        print(
            f"Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f} | val_loss: {valloss:.4f} val_rmse: {valacc:.4f}"
        )


torch.save(model.state_dict(), PATH + "mc2_808.pt")
"""plt.plot(trloss)
plt.plot(vloss)
plt.legend(["training", "test"], loc="lower right")
plt.show()"""

plt.plot(labels)
plt.plot(prediction)
plt.legend(["y_test", "pred"], loc="lower right")
plt.show()

plt.plot(t_labels)
plt.plot(t_prediction.detach().numpy())
plt.legend(["y_train", "train_pred"], loc="lower right")
plt.show()
