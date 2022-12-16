from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def neural_network_weighting(N, R, columns, number_of_splits, *args, **attributes):
    train = pd.concat([N, R])
    clf = train_neural_network(train[columns], train.label, number_of_splits)
    predictions = clf.predict_proba(N[columns].values)[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_neural_network(X_train, y_train, number_of_splits):
    features = np.shape(X_train)[1]
    param_grid = {
        "hidden_layer_sizes": [features, features // 2, features // 4],
        "learning_rate_init": [0.01, 0.001],
        "batch_size": [16, 32, 64],
    }
    nn = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        learning_rate="adaptive",
    )
    clf = GridSearchCV(nn, param_grid, cv=number_of_splits, n_jobs=-1)
    clf = clf.fit(X_train.values, y_train.values)
    return clf
