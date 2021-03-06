# !pip install yfinance
import alpaca_trade_api as tradeapi
import os    
from dotenv  import load_dotenv
from pathlib import Path
import pandas as pd
import yfinance as yf
import time
from alpha_vantage.timeseries import TimeSeries

load_dotenv()
alpha_api=os.getenv('ALPHA')
ts = TimeSeries(key='CEHZKY8UT3FCCUSH',output_format='csv')

def get_yf_stock(tickers, interval, start, end):
    """ Load stock data
    (only 2 months if intraday data needs)

    Parameter
    ----------
    tickers: name of stocks
    interval: timeframes
    start&end: datetime format

    Returns
    ----------
    yf_stocks_df : DataFrame
    """
    yf_stocks_df = yf.download(tickers = tickers, interval = interval, start = start, end = end)
    return yf_stocks_df

def get_alpha_stock(symbol, frq, timerange):
    """ Load intraday stock data

    Parameter
    ----------
    symbol: ticker of stock
    frq: timeframes
    timerange:  2 years of intraday data is evenly divided into 24 "slices" - year1month1, year1month2, year1month3, ..., year1month11, year1month12, year2month1, year2month2, year2month3, ..., year2month11, year2month12.

    Returns
    ----------
    df : DataFrame
    """
    df = []
    data, meta= ts.get_intraday_extended(symbol=symbol, interval=frq, slice=timerange)
    for row in data:
        df.append(row)
    df = pd.DataFrame(df, columns=df[0])
    df.drop(index=0,inplace=True)
    
    return df

def get_whole(symbol, frq, shares):
    """ get 7 months' share price data
    """
    df_whole = [] 
    interval = ['year1month1', 'year1month2','year1month3','year1month4','year1month5']
    for i in interval:
        df = get_alpha_stock(symbol, frq, i)
        df_whole.append(df)
    time.sleep(65)
    interval_2 = ['year1month6', 'year1month7']
    for i in interval_2:
        df2 = get_alpha_stock(symbol, frq, i)
        df_whole.append(df2)
    df_whole = pd.concat(df_whole)
    df_whole = df_whole.set_index('time').sort_index()
    return df_whole.to_csv(f'{shares}.csv')

