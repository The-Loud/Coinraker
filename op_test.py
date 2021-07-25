"""Testing the execute method before putting in the dag."""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from models.mc_dcnn import BitNet

engine = create_engine(os.getenv("src_conn_id"))

# Query the table for the prediction data
with open("sqls/prediction_data.sql", encoding="utf-8") as file:
    query = file.read()

data = pd.read_sql(query, engine)
# TODO: Use a window function on this query

# Create methods to handle the missing data points.
imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
scaler = StandardScaler()

# steps = 24  # 1 day
inp = data.drop(["load_date"], axis=1)

inp = imputer.fit_transform(inp)
inp = scaler.fit_transform(inp)

inp = torch.from_numpy(inp).float()
inp = inp.unsqueeze(0).permute(0, 2, 1)

# TODO: Verify if the split_sequence is needed
# Probably not if we only get 24 time steps back.

model = BitNet(inp.shape[1])
model.load_state_dict(torch.load("./runs/base_2.pt"))

# We don't need to track gradients for predictions.
model.eval()

output = model(inp)

print(output.item())
