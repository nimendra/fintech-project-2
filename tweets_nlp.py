# Tweet 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

 # Download/Update the VADER Lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def analyse_sentimemnt_vader(df):   
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
                "date": row['date'],
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


def process_tweets(tweet_df, timeframe):
    
    #weight the compound value by number of followers
    tweet_df['weighted_compound'] = tweet_df['compound']*tweet_df['user_followers']
    #get dummies for source
    tweet_df = pd.get_dummies(tweet_df, columns=['source'])
    #change verified into a boolean variable
    tweet_df['verified'] = tweet_df['user_verified'].apply(lambda x: 1 if (x == True) else 0)
    #resample twitter date into selected timeframe
    tweet_df = tweet_df.resample(timeframe).sum() 
    #change verified into a boolean variable
    tweet_df['verified'] = tweet_df['verified'].apply(lambda x: 1 if (x > 0) else 0)
    #drop columns
    tweet_df.drop(columns = ['compound', 'positive', 'negative', 'neutral'], inplace = True)
    # fill all missing datetime with 0
    tweet_df = tweet_df.asfreq(freq = '15min')
    
    return tweet_df