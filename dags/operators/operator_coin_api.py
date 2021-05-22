from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import List
from pycoingecko import CoinGeckoAPI
import pandas as pd


class ApiToMySql(BaseOperator):
    @apply_defaults
    def __init__(
            self,
            name: str,
            mysql_conn_id: str = None,
            tablename: str = None,
            method: str = None,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.tablename = tablename
        self.method = method

    def execute(self, context):
        cg = CoinGeckoAPI()
        call = getattr(cg, self.method)
        data = call()

        # data = cg.get_search_trending()

        df = pd.json_normalize(data['coins'])
        df['load_date'] = datetime.now()

        # Alter this to use get_connection('conn_id') and use that to get a connection.
        hook = MySqlHook(schema='source', mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        df.to_sql(self.tablename, engine, if_exists='append', index=False)

        message = f" Saving data to {self.tablename}"
        print(message)
        return message
