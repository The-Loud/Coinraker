"""
Operator for loading tweet sentiment.
Uses NLP to calculate the sentiment then saves it to a table.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from airflow.models.baseoperator import BaseOperator
from models.mc_dcnn_v2 import BitNet

# from airflow.providers.mysql.hooks.mysql import MySqlHook


class Extrapolate(BaseOperator):
    def __init__(
        self,
        *args,
        name: str = "extrapolate",
        mysql_conn_id: str = "mysql_pinwheel_source",
        table_name: str = "source.extrapolate",
        script: str = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.table_name = table_name
        self.script = script

    def pull_data(self):
        return pd.read_sql(self.script, self.mysql_conn_id)

    @staticmethod
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

    @staticmethod
    def predict(inp):

        model = BitNet(inp.shape[1])
        model.load_state_dict(torch.load("./runs/mc2_808.pt"))

        # We don't need to track gradients for predictions.
        model.eval()
        return model(inp)

    @staticmethod
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

    def save_to_table(self, frame):
        frame.to_sql(
            "source.extrapolate", self.mysql_conn_id, index=False, if_exists="replace"
        )

    def execute(self, context):
        data = self.pull_data()
        inp = self.preprocess(data)
        preds = self.predict(inp)
        df = self.build_output(data, preds)
        out = pd.concat([data, df])

        for _ in range(1):
            inp = out.tail(24)
            p_inp = self.preprocess(inp)
            preds = self.predict(p_inp)
            df = self.build_output(out, preds)
            out = pd.concat([out, df])
        out = out.reset_index()
        self.save_to_table(out)
