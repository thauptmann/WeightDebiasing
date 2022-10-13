import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold


def logistic_regression_weighting(N, R, columns, number_of_splits, *args, **attributes):
    predictions = np.zeros(len(N))
    k_fold = KFold(n_splits=number_of_splits, shuffle=True)
    for train_index, test_index in k_fold.split(N):
        train_N = N.iloc[train_index]
        test_N = N.iloc[test_index]
        train = pd.concat([train_N, R])
        clf = train_logistic_regression(train[columns], train.label)
        predictions[test_index] = clf.predict_proba(test_N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression()
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression
