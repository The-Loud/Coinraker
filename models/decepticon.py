from transformers import pipeline
import pandas as pd
from sqlalchemy import create_engine
import os

conn_str = os.environ.get('str_conn_id')

e = create_engine(conn_str)

query = "select * from tweets limit 250"

data = pd.read_sql(query, e)

# tweet = 'RT @AJA_Cortes: $BTC dropped to $3k last year and everyone was fawking crying and having meltdowns'

nlp = pipeline(task='text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
data['sen'] = data['cleansed'].apply(nlp)
print(data['sen'])

# print(f'Result: {nlp(tweet)}')