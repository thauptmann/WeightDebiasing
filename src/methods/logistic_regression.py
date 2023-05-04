from sklearn.linear_model import LogisticRegression
import pandas as pd


def logistic_regression_weighting(N, R, columns, *args, **attributes):
    train = pd.concat([N, R])
    x = train[columns].values
    y = train.label
    clf = train_logistic_regression(x, y)
    predictions = clf.predict_proba(N[columns].values)[:, 1]
    weights = (1 - predictions) / predictions
    return weights / weights.sum()


def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
