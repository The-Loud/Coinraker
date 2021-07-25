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
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        # Query the table for the prediction data
        with open(self.script, encoding="utf-8") as file:
            query = file.read()

        data = pd.read_sql(query, engine)
        # TODO: Use a window function on this query

        # Create methods to handle the missing data points.
        # steps = 24  # 1 day
        inp = data.drop(["load_date"], axis=1)

        # Create methods to handle the missing data points.
        imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        scaler = StandardScaler()

        inp = imputer.fit_transform(inp)
        inp = scaler.fit_transform(inp)

        inp = torch.from_numpy(inp).float()
        inp = inp.unsqueeze(0).permute(0, 2, 1)

        model = BitNet(inp.shape[1])
        model.load_state_dict(torch.load("./runs/base_3.pt"))

        # We don't need to track gradients for predictions.
        model.eval()

        output = model(inp)

        print(f"Prediction: {output.item()}\nActual: {data.loc[23, 'usd']}")

        data.to_sql(self.table_name, engine, if_exists="append", index=False)

        message = f"Prediction: {output.item()}\nActual: {data.loc[23, 'usd']}"
        return message
