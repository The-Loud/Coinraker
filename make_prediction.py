"""Testing the execute method before putting in the dag."""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from models.mc_dcnn_v2 import BitNet


data = pd.read_csv("data/test_set_3.csv")
data.drop_duplicates(subset="load_date")
# TODO: Use a window function on this query

# Create methods to handle the missing data points.
imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
scaler = StandardScaler()

# steps = 24  # 1 day
inp = data.set_index("load_date")

# Shift the price by one timestep
inp["prior_price"] = inp["usd"].shift(periods=1, fill_value=0)
inp.drop("usd", inplace=True, axis=1)

inp = imputer.fit_transform(inp)
inp = scaler.fit_transform(inp)

inp = torch.from_numpy(inp).float()
inp = inp.unsqueeze(0).permute(0, 2, 1)

model = BitNet(inp.shape[1])
model.load_state_dict(torch.load("./runs/mc2_804.pt"))

# We don't need to track gradients for predictions.
model.eval()
output = model(inp)

print(f"Prediction: {output.item()}\nActual: {data.loc[23, 'usd']}")
# print(f"Diff: {data.loc[23, 'usd'] - output.item()}")
