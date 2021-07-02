import tweepy
import os
import pandas as pd
import string

API_KEY = os.getenv('API_KEY')

API_SECRET_KEY = os.getenv('API_SECRET_KEY')

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)

# Things to keep
keeps = ['created_at', 'id', 'text', 'entities', 'lang']
df = pd.DataFrame()
t = tweepy.Cursor(api.search, q='bitcoin').items(500)

for tweet in t:
    if tweet.lang == 'en':
        d = pd.Series({i: getattr(tweet, i) for i in keeps})

        # This seems inefficient but for 20 rows, who cares
        df = pd.concat([df, d.to_frame().T], ignore_index=True)

def clean_tweet(tweet):
    try:
        text = []
        for c in tweet:
            if isinstance(c, int):
                text.append(c)
            elif ord(c) < 128:
                text.append(c)
            else:
                text.append('')
        return text
    except:
        return ''


# Initial preprocessing
df['text'] = df['text'].str.translate(str.maketrans('', '', string.punctuation)).str.lower()
df['text'] = df['text'].apply(lambda x: x.encode('utf-8').strip())
# df['text'] = df['text'].apply(clean_tweet)

