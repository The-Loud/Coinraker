"""
Grabs tweets and keeps only the selected fields
Sentiment is then calculated and stored in two fields
"""
import os

import pandas as pd
import tweepy
from transformers import pipeline


API_KEY = os.getenv("API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)

keeps = ["created_at", "id", "text", "entities", "lang"]
df = pd.DataFrame()
t = tweepy.Cursor(api.search, q="bitcoin").items(10)

for tweet in t:
    if tweet.lang == "en":
        d = pd.Series({i: getattr(tweet, i) for i in keeps})

        # This seems inefficient but for 20 rows, who cares
        df = pd.concat([df, d.to_frame().T], ignore_index=True)

nlp = pipeline(
    task="text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

df["text"] = df["text"].apply(lambda x: "".join([c for c in x if ord(c) < 128]))
df["sentiment"] = df["text"].apply(nlp)
df["label"] = df["sentiment"].apply(lambda x: x[0]["label"])
df["score"] = df["sentiment"].apply(lambda x: x[0]["score"])

print(df["score"].mean())
