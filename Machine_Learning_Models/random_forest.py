import pandas as pd
import datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import make_classification
from joblib import dump, load

def data_split_scale(df):
    """
    This function split and scale the features (X) and the target (y)
    """
    # Define features set
    X = df.copy()
    X.drop(["return", 'return_bol'], axis=1, inplace=True)

    # Define target vector
    y = df["return_bol"].values.reshape(-1,1)

    # Splitting into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

    # Create the StandardScaler instance
    scaler = StandardScaler()

    # Fit the Standard Scaler with the training data
    X_scaler = scaler.fit(X_train)

    # Scale the training data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def random_forests(X_train_scaled, y_train, name):
    """ running random forests classifier
    
    Parameter
    ----------
    X_train_scaled, y_train: trainning data
    name: name you want save the model


    Returns
    ----------
    predictions
    """

    # Create the random forest classifier instance
    model = RandomForestClassifier(n_estimators=500, random_state=78)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    return dump(model, f'{name}.joblib')

