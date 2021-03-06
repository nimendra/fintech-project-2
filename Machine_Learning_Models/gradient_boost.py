import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from joblib import dump, load
# Needed for decision tree visualization
import pydotplus
from IPython.display import Image



def boost_learning_rate(X_train_scaled, X_test_scaled, y_train, y_test):
    """ running gradient boost tree classifier
    
    Parameter
    ----------
    X_train_scaled, X_test_scaled, y_train, y_test

    Returns
    ----------
    learning rate choice
    """
    
    # Choose learning rate
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in learning_rates:
        model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=learning_rate,
        max_features=5,
        max_depth=3,
        random_state=0)
        model.fit(X_train_scaled,y_train.ravel())
        print("Learning rate: ", learning_rate)

        # Score the model
        print("Accuracy score (training): {0:.3f}".format(
        model.score(
            X_train_scaled,
            y_train.ravel())))
        print("Accuracy score (validation): {0:.3f}".format(
        model.score(
            X_test_scaled,
            y_test.ravel())))
        print()

    return

def gradient_boost(X_train_scaled, y_train, rate, features, name):
    """ running gradient boost tree classifier
    
    Parameter
    ----------
    X_train_scaled, y_train: trainning data
    rate = learning_rate : [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    features: int
    name: name you want save the model

    Returns
    ----------
    model
    """
    # Create GradientBoostingClassifier model
    model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=rate,
    max_features=features,
    max_depth=3,
    random_state=0)

    # Fit the model
    model.fit(X_train_scaled,y_train.ravel())

    return dump(model, f'{name}.joblib')




    
