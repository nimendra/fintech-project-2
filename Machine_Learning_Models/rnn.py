import pandas as pd
import datetime
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import dump, load


def window_data(df, window, feature_col_number, target_col_number, extra_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    Args:
           extra_col_number: the second column, from which we draw only one number - the most recent feature
    """
    X = []
    y = []
    for i in range(len(df) - window):
        features = list(df.iloc[i : (i + window), feature_col_number])
        features += [df.iloc[i + window, extra_col_number]]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

def data_split_scale(X, y):
    """
    This function split and scale the features (X) and the target (y)
    """
    # Use 70% of the data for training and the remaineder for testing
    split = int(0.7 * len(X))

    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]

    # Use the MinMaxScaler to scale data between 0 and 1.
    x_train_scaler = MinMaxScaler()
    x_test_scaler = MinMaxScaler()
    y_train_scaler = MinMaxScaler()
    y_test_scaler = MinMaxScaler()

    # Fit the scaler for the Training Data
    x_train_scaler.fit(X_train)
    y_train_scaler.fit(y_train)

    # Scale the training data
    X_train = x_train_scaler.transform(X_train)
    y_train = y_train_scaler.transform(y_train)

    # Fit the scaler for the Testing Data
    x_test_scaler.fit(X_test)
    y_test_scaler.fit(y_test)

    # Scale the y_test data
    X_test = x_test_scaler.transform(X_test)
    y_test = y_test_scaler.transform(y_test)

    # Reshape the features for the model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, y_test_scaler

def RNN(X_train, y_train , units, fraction, number, name):
    """ running LSTM model
    
    Parameter
    ----------
    X_train, y_train: trainning data
    units: neuron number
    fraction: dropout percentage
    number: epochs number
    name: name you want save the model

    Returns
    ----------
    predictions
    """
    # Define the LSTM RNN model.
    model = Sequential()

    # Initial model setup
    number_units = units
    dropout_fraction = fraction

    # Layer 1
    model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
    model.add(Dropout(dropout_fraction))

    # Layer 2
    model.add(LSTM(units=number_units, return_sequences=True))
    model.add(Dropout(dropout_fraction))

    # Layer 3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Summarize the model
    model.summary()

    #fit model
    model.fit(X_train, y_train, epochs=number, shuffle=False, batch_size=90, verbose=1)

    # Save model as JSON
    model_json = model.to_json()
    file_path = Path(f"{name}.json")
    with open(file_path, "w") as json_file:
        json_file.write(model_json)

    # Save weights
    file_path = f"{name}.h5"
    model.save_weights(file_path)

    return 