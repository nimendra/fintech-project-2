import tweepy
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.util import ngrams
import re
import pandas as pd

def generate_wordcloud(df, n):
    my_dpi = 1
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words('english'))
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', ' '.join(df['text']))
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]

    sw_words = [word.lower() for word in lem if word.lower() not in sw]
    mgrams = ngrams(sw_words, n)

    output = ['_'.join(i) for i in mgrams]
    all_text = ' '.join(output)
    wc = WordCloud(width=1080, height=720).generate(all_text)
    wc.to_file(f"new_tweets_wordCloud_{n}grams.png")
    return wc

consumer_key = 'SCOPBRKeG4nRCEa7XkMoqQ'
consumer_secret = 'RYInMkLiNyg0iKC3g89Y0f8g8kbNFSsNjZXNYBYILQ'
access_token = '107275871-PtAa9t7OJX82IEbGjdOzaFMgeCWQyU8haNPhN4mD'
access_token_secret = 'Q691rTrztjXeYAP5FEA5kZWRhuFRtH8sq6msPgUeD1TzI'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

# csvFile = open('./Resources/new_tweets.csv', 'w')
# csvWriter = csv.writer(csvFile)
# # Write the header row
# csvWriter.writerow(['created_at', 'text'])

# all_tweets = []

# for tweet in tweepy.Cursor(api.search,q="#covidvaccine",count=1000,
#                            lang="en",
#                            since="2019-11-01").items(1000):
#     print (tweet.created_at, tweet.text)
#     all_tweets.append(tweet.text)
#     csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

new_tweets_df = pd.read_csv('./Resources/new_tweets.csv')
generate_wordcloud(new_tweets_df, 2)