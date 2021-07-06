import preprocessor as p
import pandas as pd
from sqlalchemy import create_engine
import os

conn_str = os.environ.get('src_conn_id')

e = create_engine(conn_str)

file = "../sqls/pull_tweets.sql"

with open(file, 'r') as f:
    query = f.read()

data = pd.read_sql(query, e)

data['cleansed'] = data['text'].apply(p.clean)
print(data.head())

data.to_sql('tweets', e, if_exists='replace', index=False)