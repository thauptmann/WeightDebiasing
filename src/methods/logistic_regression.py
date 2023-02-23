from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def logistic_regression_weighting(N, R, columns, number_of_splits, *args, **attributes):
    train = pd.concat([N, R])
    clf = train_logistic_regression(train[columns], train.label)
    predictions = clf.predict_proba(N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    weights = weights.numpy().astype(np.float64)
    return weights / weights.sum()


def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression()
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
