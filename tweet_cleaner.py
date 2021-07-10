"""
Preliminary attempt at cleaning tweets
May not need this file
"""
import os

import pandas as pd
import preprocessor as p
from sqlalchemy import create_engine

conn_str = os.environ.get("src_conn_id")

e = create_engine(conn_str)

FILE = "../sqls/pull_tweets.sql"

with open(FILE, "r") as f:
    query = f.read()

data = pd.read_sql(query, e)

data["cleansed"] = data["text"].apply(p.clean)
print(data.head())

data.to_sql("tweets", e, if_exists="replace", index=False)
