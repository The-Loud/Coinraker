"""
Operator for loading tweet sentiment.
Uses NLP to calculate the sentiment then saves it to a table.
"""
import pandas as pd
from airflow.models.baseoperator import BaseOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from transformers import pipeline


class TweetSentiment(BaseOperator):
    """
    This operator will pull tweets at time t and convert them to a sentiment rating and score.
    The sentiment rating is then transformed via ordinal encoder and then averaged
    """

    def __init__(
        self,
        *args,
        name: str = None,
        mysql_conn_id: str = None,
        table_name: str = "sentiment",
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
        Pull all tweets for time t.
        Run sentiment Analysis.
        Put tweets back into table...or process to a new table. probably this one.
        :param context:
        :return:
        """

        # Build the connection
        hook = MySqlHook(schema="source", mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        # Create the BERT NLP classifier
        nlp = pipeline(
            task="text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )

        # Pull data from the DB for tweets at time t and compute sentiment
        with open(self.script, encoding="utf-8") as file:
            query = file.read()

        data = pd.read_sql(query, engine)
        data["sentiment"] = data["cleansed"].apply(nlp)
        data["label"] = data["sentiment"].apply(lambda x: x[0]["label"])
        data["score"] = data["sentiment"].apply(lambda x: x[0]["score"])
        data.drop("sentiment", inplace=True)

        data.to_sql(self.table_name, engine, if_exists="append", index=False)

        message = f" Saving data to {self.table_name}"
        print(message)
        return message
