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


class PredictPrice(BaseOperator):
    """
    The price prediction operator. This loads the model and calculates t+1 using n-23 (24) steps.
    """

    def __init__(
        self,
        *args,
        name: str = None,
        mysql_conn_id: str = "mysql_pinwheel_source",
        table_name: str = "predictions",
        script: str = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.table_name = table_name
        self.script = script

    def execute(self, context):
        """
        1. Pull data into a dataframe
        2. Load up model
        3. Calculate the prediction
        4. Calculate loss
        5. Store load_date, loss, actual, and predicted
        """

        # Build the connection
        engine = self.mysql_conn_id
        # hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        # engine = hook.get_sqlalchemy_engine()

        # Query the table for the prediction data
        with open("sqls/prediction_data.sql", encoding="utf-8") as file:
            query = file.read()

        data = pd.read_sql(query, engine)
        # data.drop_duplicates(subset="load_date")
        data["prior_price"] = data["usd"]  # .shift(periods=1, fill_value=0)

        # Create methods to handle the missing data points.
        imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        scaler = StandardScaler()

        # steps = 24  # 1 day
        inp = data.set_index("load_date")

        # Shift the price by one timestep
        inp.drop("usd", inplace=True, axis=1)

        inp = imputer.fit_transform(inp)
        inp = scaler.fit_transform(inp)

        inp = torch.from_numpy(inp).float()
        inp = inp.unsqueeze(0).permute(0, 2, 1)

        model = BitNet(inp.shape[1])
        model.load_state_dict(torch.load("./runs/mc2_808.pt"))

        # We don't need to track gradients for predictions.
        model.eval()
        output = model(inp)

        final = data.loc[23].copy()
        final["prediction"] = final["prior_price"] + output.item()
        final["diff"] = final["prediction"] - final["usd"]

        final = final.to_frame().T  # .reset_index().drop("index", axis=1)
        print(final)

        # Concatenate the prediction and the data and put it in a new table.
        data.to_sql(self.table_name, engine, if_exists="append", index=False)

        # message = f"Prediction: {output.item()}\nActual: {data.loc[23, 'usd']}"
        message = output.item() / data.loc[23, "usd"]
        return message
