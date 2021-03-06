import pandas as pd
import datetime
from pathlib import Path
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download/Update the VADER Lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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
    df = df[df.date.notnull()]
    df = df[df.date.str.contains("-")]

    
    # Parse and set Date as the index
    df['date'] = df['date'].apply(lambda t: try_parse(t))
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df = df.astype({'user_followers': int, 'user_friends': int, 'user_favourites': int})
    df = df.dropna()

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

def process_tweets(ret_df, timeframe):
    """ Cleaning tweets data and resample it into intraday interval
    Parameter
    ----------
    ret_df : DateFrame
    timeframe: frequency values: 15min, 30min, 60min

    Returns
    ----------
    tweet_df : DataFrame grouped into trading hour and after hour and resampled
    """
    
    #weight the compound value by number of followers
    ret_df['weighted_compound'] = ret_df['compound']*ret_df['user_followers']
    #get dummies for source
    ret_df = pd.get_dummies(ret_df, columns=['source'])
    #change verified into a boolean variable
    ret_df['verified'] = ret_df['user_verified'].apply(lambda x: 1 if (x == True) else 0)
    #resample twitter date into selected timeframe
    ret_df = ret_df.resample(timeframe).sum()
    #change verified into a boolean variable
    ret_df['verified'] = ret_df['verified'].apply(lambda x: 1 if (x > 0) else 0)
    #drop columns
    ret_df.drop(columns = ['compound', 'positive', 'negative', 'neutral'], inplace = True)
    # fill all missing datetime with 0
    ret_df = ret_df.asfreq(freq = timeframe)
    #slice trading our tweets 
    tweet_df_trade = ret_df.between_time('04:15','20:00')
    #group after hour tweets by day
    tweet_df_afh = ret_df.between_time('20:15','04:00').resample('d').sum()
    #join two together 
    tweet_df = pd.concat([tweet_df_trade, tweet_df_afh], axis=0, join='outer').sort_index()
    
    return tweet_df

def load_stock_csv(fileName):
    """ Load tweet data

    Parameter
    ----------
    fileName : string

    Returns
    ----------
    df : DataFrame
    """
    share_df = pd.read_csv(f'./Resources/{fileName}.csv', parse_dates=True, index_col='time',)
    share_df.index.names = ['date']
    
    return share_df

def get_return(share_df):
    """ calculate returns from close prices

    Parameter
    ----------
    share_df : DateFrame

    Returns
    ----------
    share_return_df : DataFrame
    """
    share_return_df = share_df[['close']].pct_change().dropna()
    share_return_df.rename(columns={'close': 'return'}, inplace = True)

    return share_return_df

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def concat_df(tweet_df, share_return_df):
    """ combine two dataframes
    """

    combined_df = pd.concat([tweet_df, share_return_df], axis=1, join='outer')
    combined_df.dropna(subset=['weighted_compound'], inplace=True)

    return combined_df 

def changeReturn(returns):
    """ convert returns to boolean value
    """
    if returns > 0:
        return 1
    else:
        return 0

def final_df(combined_df, lag):
    """ combine tweets data with lagged share returns

    Parameter
    ----------
    combined_df : DateFrame
    lag: int

    Returns
    ----------
    combined_df : DataFrame
    """
    
    combined_df['return'] = combined_df['return'].shift(lag)
    combined_df['return'].fillna(method= 'backfill', inplace=True)
    combined_df.dropna(inplace=True)
    combined_df['return_bol'] = combined_df['return'].apply(changeReturn)

    return combined_df


