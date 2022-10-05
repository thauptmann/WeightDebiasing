from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def cv_bootstrap_prediction(df, columns, number_of_splits):
    clf = train_forest(df[columns], df.label, df["weights"])
    predictions = clf.predict_proba(df[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_forest(X_train, y_train, number_of_splits):
    param_grid = []
    forest = RandomForestClassifier(max_depth=5, n_estimators=50)
    decision_stump = forest.fit(X_train, y_train)
    return decision_stump
