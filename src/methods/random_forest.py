from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np

param_grid = {"max_depth": [2, 3, 5], "n_estimators": [25, 50, 100]}


def random_forest_weighting(N, R, columns, number_of_splits, *args, **kwargs):
    predictions = np.zeros(len(N))
    k_fold = KFold(n_splits=number_of_splits, shuffle=True)
    for train_index, test_index in k_fold.split(N):
        train_N = N.iloc[train_index]
        test_N = N.iloc[test_index]
        train = pd.concat([train_N, R])
        clf = train_forest(train[columns], train.label, int(number_of_splits / 2))
        predictions[test_index] = clf.predict_proba(test_N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_forest(X_train, y_train, number_of_splits):
    forest = RandomForestClassifier()
    clf = GridSearchCV(forest, param_grid, cv=number_of_splits, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    return clf
