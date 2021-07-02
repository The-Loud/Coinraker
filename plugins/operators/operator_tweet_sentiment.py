from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from transformers import pipeline
import pandas as pd


class TweetSentiment(BaseOperator):
    """
    This operator will pull tweets at time t and convert them to a sentiment rating and score.
    The sentiment rating is then transformed via ordinal encoder and then averaged
    """

    @apply_defaults
    def __init__(
            self,
            name: str,
            mysql_conn_id: str = None,
            tablename: str = 'sentiment',
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.mysql_conn_id = mysql_conn_id
        self.tablename = tablename

    def execute(self, context):
        """
        Pull all tweets for time t.
        Run sentiment Analysis.
        Put tweets back into table...or process to a new table. probably this one.
        :param context:
        :return:
        """

        # Build the connection
        hook = MySqlHook(schema='source', mysql_conn_id=self.mysql_conn_id)
        engine = hook.get_sqlalchemy_engine()

        # Create the BERT NLP classifier
        nlp = pipeline(task='text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')

        # Pull data from the DB for tweets at time t and compute sentiment
        with open('./sqls/pull_tweets.sql', encoding='utf-8') as q:
            query = q.read()

        df = pd.read_sql(query, engine)
        df['sentiment'] = df['text'].apply(lambda x: nlp(x))
        df['label'] = df['sentiment'].apply(lambda x: x[0]['label'])
        df['score'] = df['sentiment'].apply(lambda x: x[0]['score'])
        df.drop('sentiment', inplace=True)

        df['load_date'] = datetime.now()
        df.to_sql(self.tablename, engine, if_exists='append', index=False)

        message = f" Saving data to {self.tablename}"
        print(message)
        return message
