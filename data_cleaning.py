import pandas as pd
import datetime
#load in vaccine dataframe, takes in the filepath of the vaccine csv
def load_vaccine_csv(path):
    
    df = pd.read_csv(path)
    df=df[['user_name','user_location','user_followers','user_friends','user_favourites','user_verified','date','text','source']]
    
    # '18-08-2020 12:55'
    # df.index = pd.to_datetime(df['date'])
    # for managebility we remove the hashtag, but it might have a future use-case
    # is_retweet is all False in this dataset
    # df = df[df['is_retweet'] == False]
    df = df.dropna()
    return df

#load in stock dataframe, takes in the filepath of the stock csv
def load_stock_csv(path):
    df = pd.read_csv('./Resources/Pfizer.csv', parse_dates=True, index_col='time',)
    df.index.names = ['date']
    return df


    
def try_parse(dateString):
    """ Try and parse a Date string in 4 different formats
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

            

        
#Function to parse the dates in the twitter dataframe and set/sort the index      
def process_twitter_dataframe(df):
    df = df[df.date.notnull()]
    df = df[df.date.str.contains("-")]

    df['date'] = df['date'].apply(lambda t: try_parse(t))
    df['xxx'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

    df = df.sort_index()
    df = df.astype({'user_followers': int, 'user_friends': int, 'user_favourites': int})
    df2 = df.dropna()
    return df

#process the stock dataframe, add a returns column, shift the data and return a dataframe containing the return
def process_stock_dataframe(df, shift):
    df = df[['close']].pct_change().dropna()
    df.rename(columns={'close': 'return'}, inplace = True)
    df = df.shift(shift)
    df.dropna(inplace=True)
    return df

#change the returns to -1 or 1
def changeReturn(returns):
    if returns > 0:
        return 1
    elif returns < 0:
        return -1
    else:
        return 0
    
