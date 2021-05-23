from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import List
import pandas as pd
import tweepy
import os


class TweetToMySql(BaseOperator):
    @apply_defaults
    def __init__(
            self,
            name: str,
            mysql_conn_id: str = None,
            tablename: str = None,
            search_query: str = None,
            item_count: int = 20,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.tablename = tablename
        self.search_query = search_query
        self.item_count = item_count

    def execute(self, context):

        API_KEY = os.getenv('API_KEY')
        API_SECRET_KEY = os.getenv('API_SECRET_KEY')

        auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
        api = tweepy.API(auth)

        # Things to keep
        keeps = ['created_at', 'id', 'text', 'entities', 'lang']
        df = pd.DataFrame()
        t = tweepy.Cursor(api.search, q=self.search_query).items(self.item_count)

        for tweet in t:
            d = pd.Series({i: getattr(tweet, i) for i in keeps})

            # This seems inefficient but for 20 rows, who cares
            df = pd.concat([df, d.to_frame().T], ignore_index=True)

        # df = pd.json_normalize(json_tweets)
        df['load_date'] = datetime.now()

        # Alter this to use get_connection('conn_id') and use that to get a connection.
        hook = MySqlHook(schema='source', mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        df.to_sql(self.tablename, engine, if_exists='append', index=False)

        message = f" Saving data to {self.tablename}"
        print(message)
        return message
