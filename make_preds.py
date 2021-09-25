import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from models.mc_dcnn_v2 import BitNet


engine = create_engine(os.getenv("src_conn_id"))


def pull_data():
    with open("sqls/prediction_data.sql", encoding="utf-8") as file:
        query = file.read()

    return pd.read_sql(query, engine)


def preprocess(data):
    data["prior_price"] = data["usd"]

    # Create methods to handle the missing data points.
    imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    scaler = StandardScaler()

    # steps = 24  # 1 day
    inp = data.set_index("load_date")

    # Shift the price by one timestep
    try:
        inp.drop(["usd", "prediction", "diff"], inplace=True, axis=1)
    except KeyError:
        inp.drop(["usd"], inplace=True, axis=1)

    inp = imputer.fit_transform(inp)
    inp = scaler.fit_transform(inp)

    inp = torch.from_numpy(inp).float()
    inp = inp.unsqueeze(0).permute(0, 2, 1)

    return inp


def predict(inp):

    model = BitNet(inp.shape[1])
    model.load_state_dict(torch.load("./runs/mc2_808.pt"))

    # We don't need to track gradients for predictions.
    model.eval()
    return model(inp)


def build_output(data, output):

    final = data.iloc[-1].copy()
    final["prediction"] = final["prior_price"] + output.item()
    final["diff"] = final["prediction"] - final["usd"]
    final["usd"] = final["prediction"]
    final["usd_24h_change"] = data["usd_24h_change"].mean()
    final["usd_24h_vol"] = data["usd_24h_vol"].mean()
    final["usd_market_cap"] = data["usd_market_cap"].mean()
    final["avg_rating"] = data["avg_rating"].mean()
    final["load_date"] = final.load_date + pd.Timedelta(hours=1)

    final = final.to_frame().T
    return final


def save_to_table(frame):
    frame.to_sql("source.extrapolate", engine, index=False, if_exists="replace")


def main():
    data = pull_data()
    inp = preprocess(data)
    preds = predict(inp)
    df = build_output(data, preds)
    out = pd.concat([data, df])

    for _ in range(1):
        inp = out.tail(24)
        p_inp = preprocess(inp)
        preds = predict(p_inp)
        df = build_output(out, preds)
        out = pd.concat([out, df])
    out = out.reset_index()
    save_to_table(out)

    plt.plot(out["load_date"], out["prediction"])
    plt.show()


if __name__ == "__main__":
    main()
