# Tweet 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dotenv import load_dotenv
import numpy as np
import hvplot.pandas
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.util import ngrams
import re

def try_parse(dateString):
    """ Try and parse a Date string in 4 possible different formats

    Parameter
    ----------
    dateString : string

    Returns
    ----------
    date : date
    """
    # return dateString
    # print(dateString)
    try:
        return datetime.datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        try:
            return datetime.datetime.strptime(dateString, '%Y-%m-%d %H:%M')
        except Exception as e:
            try:
                return datetime.datetime.strptime(dateString, '%d-%m-%Y %H:%M')
            except Exception as e:
                return datetime.datetime.strptime(dateString, '%d-%m-%Y %H:%M:%S')

def load_tweets_csv(fileName):
    """ Load tweet data

    Parameter
    ----------
    fileName : string

    Returns
    ----------
    df : DataFrame
    """
    df = pd.read_csv(f'./Resources/{fileName}.csv')
    df = df[['user_name', 'user_location', 'user_followers', 'user_friends', 'user_favourites', 'user_verified', 'date', 'text', 'source']]
    return df

def analyse_sentimemnt_vader(df):   
    """ Analyse sentiments with Vader

    Parameter
    ----------
    df : DateFrame

    Returns
    ----------
    ret_df : DataFrame with negative, positive and neutral scores
    """
    sentiments = []

    for index, row in df.iterrows():
        try:
            text = row["text"]           
            sentiment = analyzer.polarity_scores(text)
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]
            
            sentiments.append({
                "text": text,
                "date": index,
                "user_location":row['user_location'],
                "user_followers":row['user_followers'],
                "user_friends":row['user_friends'],
                "user_favourites":row['user_favourites'],
                "user_name": row['user_name'],
                "user_verified": row['user_verified'],
                "source": row['source'],
                "compound": compound,
                "positive": pos,
                "negative": neg,
                "neutral": neu           
            })
            
        except AttributeError:
            pass
        
    # Create DataFrame
    ret_df = pd.DataFrame(sentiments)

    # Reorder DataFrame columns
    cols = ["date","user_name", "user_location","user_followers", "user_friends", "user_favourites", "text", "compound", "positive", "negative", "neutral", "user_verified", "source"]
    ret_df = ret_df[cols]
    ret_df.set_index('date', inplace=True)
    ret_df = ret_df.sort_index()

    return ret_df

def generate_wordcloud(df, n):
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
    wc.to_file(f"wordCloud_{n}grams.png")
    return wc

def plot(df):
    d2 = df.resample('M').mean()
    # print(d2)
    bar1 = d2.hvplot(
                        y=["positive", "negative", "neutral"],
                        kind="bar",
                        height=400, width=800,
                        ylabel="Positive, Negative and Neutral Sentiments",
                        title="Sentiments",
                        rot=90)
    return bar1


# Load Tweets
df = load_tweets_csv('tweets')
df = df[df.date.notnull()]
df = df[df.date.str.contains("-")]

# Parse and set Date as the index
df['date'] = df['date'].apply(lambda t: try_parse(t))
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
df.set_index('date', inplace=True)
df = df.sort_index()

print(df.head())

# Vader NLTK
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

ret_df = analyse_sentimemnt_vader(df)

# Save the new Data Frame
saved_df = ret_df[['user_name','user_location','user_followers','user_friends','user_favourites','user_verified','source', 'compound','positive', 'negative','neutral']]
print(saved_df)
saved_df.to_csv('./Resources/tweets_with_sentiment.csv')

plt = plot(ret_df)
# hvplot.show(plt)
hvplot.save(plt, 'tweets_sentiment_resampled.png')

wc = generate_wordcloud(df, 1)
wc = generate_wordcloud(df, 2)
wc = generate_wordcloud(df, 3)
