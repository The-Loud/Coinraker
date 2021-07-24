"""
Operator for loading tweet sentiment.
Uses NLP to calculate the sentiment then saves it to a table.
"""
import numpy as np
import pandas as pd
import torch
from airflow.models.baseoperator import BaseOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from models.mc_dcnn import BitNet


class PredictPrice(BaseOperator):
    """
    The price prediction operator. This loads the model and calculates t using n-23 (24) steps.
    """

    def __init__(
        self,
        *args,
        name: str = None,
        mysql_conn_id: str = "src_conn_id",
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
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        # Query the table for the prediction data
        with open(self.script, encoding="utf-8") as file:
            query = file.read()

        data = pd.read_sql(query, engine)
        # TODO: Use a window function on this query

        # Create methods to handle the missing data points.
        imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        scaler = StandardScaler()

        # steps = 24  # 1 day
        inp = data.drop(["load_date", "crypto", "id"], axis=1)

        inp = imputer.fit_transform(inp)
        inp = scaler.fit_transform(inp)

        inp = inp.permute(0, 2, 1)

        # TODO: Verify if the split_sequence is needed.
        # Probably not if we only get 24 time steps back.

        model = BitNet(inp.shape[1])
        model.load_state_dict(torch.load("../runs/base.pt"))

        # We don't need to track gradients for predictions.
        model.eval()

        data["predicted_price"] = model(inp)

        data.to_sql(self.table_name, engine, if_exists="append", index=False)

        message = f" Saving data to {self.table_name}"
        print(message)
        return message
