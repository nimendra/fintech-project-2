import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

def fetch_data():
    """Fetches data for algo trading"""
    # add close price to combined_reg_df

    algo_df = pd.concat([combined_reg_df[['weighted_compound', 'return']], share_df[['close']]], join = 'inner', axis = 1)

    # Print the DataFrame
    print(algo_df.head())
    return algo_df

def generate_signals(algo_df, short, long, bollinger):
    """Generates trading signals for a given dataset.

    Parameter
    ----------
    algo_df: dataframe
    long, short, bollinger: windows for MA and Bollinger Band
    
    """

    # calculate trading signal based on compound score
    algo_df['weighted_compound_long'] = np.where(algo_df['weighted_compound'] > 0.05,1.0 , 0.0)
    algo_df['weighted_compound_short'] = np.where(algo_df['weighted_compound'] < -0.05, -1.0, 0.0)
    algo_df['sentiment_signal'] = algo_df['weighted_compound_long'] + algo_df['weighted_compound_short']

    # Set short and long windows
    short_window = short
    long_window = long

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    algo_df['fast_close'] = algo_df['close'].ewm(halflife=short_window).mean()
    algo_df['slow_close'] = algo_df['close'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    algo_df['crossover_long'] = np.where(algo_df['fast_close'] > algo_df['slow_close'], 1.0, 0.0)
    algo_df['crossover_short'] = np.where(algo_df['fast_close'] < algo_df['slow_close'], -1.0, 0.0)
    algo_df['crossover_signal'] = algo_df['crossover_long'] + algo_df['crossover_short']

    # Set bollinger band window
    bollinger_window = bollinger

    # Calculate rolling mean and standard deviation
    algo_df['bollinger_mid_band'] = algo_df['close'].rolling(window=bollinger_window).mean()
    algo_df['bollinger_std'] = algo_df['close'].rolling(window=bollinger_window).std()

    # Calculate upper and lowers bands of bollinger band
    algo_df['bollinger_upper_band']  = algo_df['bollinger_mid_band'] + (algo_df['bollinger_std'] * 1)
    algo_df['bollinger_lower_band']  = algo_df['bollinger_mid_band'] - (algo_df['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    algo_df['bollinger_long'] = np.where(algo_df['close'] < algo_df['bollinger_lower_band'], 1.0, 0.0)
    algo_df['bollinger_short'] = np.where(algo_df['close'] > algo_df['bollinger_upper_band'], -1.0, 0.0)
    algo_df['bollinger_signal'] = algo_df['bollinger_long'] + algo_df['bollinger_short']

    return algo_df

def prepare_model(algo_df):
    """ 
    prepare data of random forest model for trading signal
    """

    # Set x variable list of features
    x_var_list = ['sentiment_signal', 'crossover_signal', 'bollinger_signal']

    # Shift DataFrame values by 1
    algo_df[x_var_list] = algo_df[x_var_list].shift(1)

    # Drop NAs and replace positive/negative infinity values
    algo_df.dropna(subset=x_var_list, inplace=True)
    algo_df.dropna(subset=['return'], inplace=True)
    algo_df = algo_df.replace([np.inf, -np.inf], np.nan)

    # Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
    algo_df['Positive Return'] = np.where(algo_df['return'] > 0, 1.0, 0.0)

    # Construct training start and end dates
    training_start = algo_df.index.min().strftime(format= "%Y-%m-%d %H:%M:%S")
    training_end = '2021-01-01 20:00:00'

    # Construct testing start and end dates
    testing_start =  '2021-01-02 04:15:00'
    testing_end = algo_df.index.max().strftime(format= "%Y-%m-%d %H:%M:%S")

    # Print training and testing start/end dates
    print(f"Training Start: {training_start}")
    print(f"Training End: {training_end}")
    print(f"Testing Start: {testing_start}")
    print(f"Testing End: {testing_end}")

    # Construct the x train and y train datasets
    X_train = algo_df[x_var_list][training_start:training_end]
    y_train = algo_df['Positive Return'][training_start:training_end]

    # Construct the x test and y test datasets
    X_test = algo_df[x_var_list][testing_start:testing_end]
    y_test = algo_df['Positive Return'][testing_start:testing_end]

    return X_train, X_test, y_train, y_test 

def random_forests(X_train, y_train, name):
    """ running random forests classifier
    
    Parameter
    ----------
    X_train_scaled, y_train: trainning data
    name: name you want save the model

    Returns
    ----------
    predictions
    """
    # Fit a SKLearn linear regression using just the training set (X_train, Y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=78)
    model = model.fit(X_train, y_train)

    return dump(model, f'{name}.joblib')