from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np

param_grid = {"max_depth": [2, 3, 5], "n_estimators": [25, 50, 100]}


def random_forest_weighting(N, R, columns, number_of_splits, *args, **kwargs):
    train = pd.concat([N, R])
    clf = train_forest(train[columns], train.label, number_of_splits)
    predictions = clf.predict_proba(N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_forest(X_train, y_train, number_of_splits):
    forest = RandomForestClassifier()
    clf = GridSearchCV(forest, param_grid, cv=number_of_splits, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    return clf
