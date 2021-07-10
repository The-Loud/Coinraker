"""
Test module for the NLP tool
"""
import os

import pandas as pd
from sqlalchemy import create_engine
from transformers import pipeline

conn_str = os.environ.get("str_conn_id")

e = create_engine(conn_str)

QUERY = "select * from tweets limit 250"

data = pd.read_sql(QUERY, e)

# tweet = 'dropped to $3k last year and everyone was fawking crying and having meltdowns'

nlp = pipeline(
    task="text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment"
)
data["sen"] = data["cleansed"].apply(nlp)
print(data["sen"])

# print(f'Result: {nlp(tweet)}')
