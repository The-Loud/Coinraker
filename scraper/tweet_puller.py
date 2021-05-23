import tweepy
import os


API_KEY = os.getenv('API_KEY')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET_KEY)
api = tweepy.API(auth)

for tweet in tweepy.Cursor(api.search, q='$BTC').items(50):
    print(tweet.text)

