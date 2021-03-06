import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load

def gnb(X_train_scaled, y_train, name):
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
    model = GaussianNB()
    
    #fit model
    model.fit(X_train_scaled, y_train)


    return dump(model, f'{name}.joblib')