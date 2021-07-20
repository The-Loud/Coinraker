"""
Main operator to load tweets to MySQL.
Runs against Twitter API and formats the data
Loads data into the database
"""
import string
from datetime import datetime

import pandas as pd
import preprocessor as p
import tweepy
from airflow.models import Variable
from airflow.models.baseoperator import BaseOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook


class TweetToMySql(BaseOperator):
    """
    MYSQL tweet loader
    """

    def __init__(
        self,
        *args,
        name: str,
        mysql_conn_id: str = None,
        table_name: str = None,
        search_query: str = None,
        item_count: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.table_name = table_name
        self.search_query = search_query
        self.item_count = item_count

    def execute(self, context):

        api_key = Variable.get("API_KEY")
        api_secret_key = Variable.get("API_SECRET_KEY")

        auth = tweepy.AppAuthHandler(api_key, api_secret_key)
        api = tweepy.API(auth)

        # Things to keep
        keeps = ["created_at", "id", "text", "entities", "lang"]
        data = pd.DataFrame()
        tweets = tweepy.Cursor(api.search, q=self.search_query).items(self.item_count)

        for tweet in tweets:
            if tweet.lang == "en":
                series = pd.Series({i: getattr(tweet, i) for i in keeps})

                # This seems inefficient but for 20 rows, who cares
                data = pd.concat([data, series.to_frame().T], ignore_index=True)

        # df = pd.json_normalize(json_tweets)
        data["load_date"] = datetime.now()

        # Initial preprocessing
        data["text"] = data["text"].str.translate(
            str.maketrans("", "", string.punctuation)
        )  # .str.lower()
        data["text"] = data["text"].apply(lambda x: x.encode("utf-8").strip())
        data.drop("entities", inplace=True, axis=1)

        # Clean up basic garbage from the tweets
        data["cleansed"] = data["text"].apply(p.clean)

        # Alter this to use get_connection('conn_id') and use that to get a connection.
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        data.to_sql(self.table_name, engine, if_exists="replace", index=False)

        message = f" Saving data to {self.table_name}"
        return message
