"""
The training function for the network. handles data preprocessing and model training.
"""
import numpy as np
import pandas as pd
import torch
from airflow.models.baseoperator import BaseOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.mc_dcnn import BitNet
from utils import split_sequence


PATH = "./runs/"
se = SimpleImputer(strategy="mean", missing_values=np.nan)
ss = StandardScaler()


class RetrainModel(BaseOperator):
    """
    Retrains the model should the accuracy fall below a certain place. Uses Transfer learning and
    the last week of data (7 x 24 = 168 observations).
    """

    def __init__(
        self,
        *args,
        name: str = None,
        mysql_conn_id: str = "mysql_pinwheel_source",
        table_name: str = "predictions",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.table_name = table_name

    def execute(self, context):

        # Build the connection
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        # Import dataset
        data = pd.read_sql_table(
            table_name=self.table_name, con=engine, schema="source"
        )
        data = data.drop_duplicates(subset="load_date")

        data.set_index("load_date", inplace=True)
        data.dropna(inplace=True)

        # Shift the price by one timestep
        data["prior_price"] = data["usd"].shift(periods=1, fill_value=0)

        y = data["usd"]
        X = data.drop(["crypto", "id", "AVG(s2.score)", "row_num", "usd"], axis=1)

        X = se.fit_transform(X)
        X = ss.fit_transform(X)

        y = y.to_numpy()

        # Split the data into 24-hour subsequences
        X, y = split_sequence(X, y, 24)

        X = X.permute(0, 2, 1)

        # Load up the original weights for training
        model = BitNet(X.shape[1])
        model.load_state_dict(torch.load("./runs/base_731.pt"))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train model with model.train()
        model.train()
        for epoch in range(18):
            running_loss = 0.0
            for i, value in enumerate(X):
                inputs = value.unsqueeze(0)
                labels = y[i]
                prediction = model(inputs)
                loss = criterion(prediction.squeeze(), labels)
                running_loss += loss.item() * inputs.size(0)
                optimizer.zero_grad()
                loss.backward()  # this is backpropagation to calculate gradients
                optimizer.step()  # applying gradient descent to update weights and bias values

            print(
                "epoch: ", epoch, " loss: ", np.sqrt(running_loss / len(X))
            )  # print out loss for each epoch

        torch.save(model.state_dict(), PATH + "base_731.pt")

        message = np.sqrt(running_loss / len(X))
        return message
