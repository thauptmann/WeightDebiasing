import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


def logistic_regression_weighting(df, columns, **attributes):
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    predictions = np.zeros(len(N))
    train = pd.concat([N, R])
    clf = train_logistic_regression(train[columns], train.label)
    predictions = clf.predict_proba(N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression()
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
