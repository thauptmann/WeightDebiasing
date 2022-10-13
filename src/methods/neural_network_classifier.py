from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np


def neural_network_weighting(N, R, columns, number_of_splits, *args, **attributes):
    predictions = np.zeros(len(N))
    k_fold = KFold(n_splits=number_of_splits, shuffle=True)
    for train_index, test_index in k_fold.split(N):
        train_N = N.iloc[train_index]
        test_N = N.iloc[test_index]
        train = pd.concat([train_N, R])
        clf = train_neural_network(
            train[columns], train.label, int(number_of_splits / 2)
        )
        predictions[test_index] = clf.predict_proba(test_N[columns].values)[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_neural_network(X_train, y_train, number_of_splits):
    features = np.shape(X_train)[1]
    param_grid = {
        "hidden_layer_sizes": [4, 8, int(features / 2)],
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
