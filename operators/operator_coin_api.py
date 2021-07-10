"""
Operator for pulling coinGecko data.
The operator takes in a set of coins and will make the call
then load the data into MySQL.
"""
from datetime import datetime
from typing import List

import pandas as pd
from airflow.models.baseoperator import BaseOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from pycoingecko import CoinGeckoAPI


class ApiToMySql(BaseOperator):
    """
    Custom operator to use the MySQL connection to CoinGecko
    """

    def __init__(
        self,
        *args,
        name: str,
        mysql_conn_id: str = None,
        table_name: str = None,
        method: str = None,
        coins: List[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.table_name = table_name
        self.method = method
        self.coins = coins

    def execute(self, context):
        coin_gecko = CoinGeckoAPI()
        call = getattr(coin_gecko, self.method)
        if self.method == "get_price":
            data = call(self.coins, "usd")
        else:
            data = call()

        data = pd.json_normalize(data["coins"])
        data["load_date"] = datetime.now()

        # Alter this to use get_connection('conn_id') and use that to get a connection.
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        data.to_sql(self.table_name, engine, if_exists="append", index=False)

        message = f" Saving data to {self.table_name}"
        print(message)
        return message
