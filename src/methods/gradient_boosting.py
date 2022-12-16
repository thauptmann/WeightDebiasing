import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    "learning_rate": [0.1, 0.01],
    "n_estimators": [100, 200],
    "max_depth": [1, 3],
}


def gradient_boosting_weighting(N, R, columns, number_of_splits, *args, **kwargs):
    train = pd.concat([N, R])
    clf = train_gradient_boosting(
        train[columns], train.label, int(number_of_splits / 2)
    )
    predictions = clf.predict_proba(N[columns])[:, 1]
    weights = (1 - predictions) / predictions
    return weights


def train_gradient_boosting(X_train, y_train, number_of_splits):
    gradient_boosting = GradientBoostingClassifier()
    clf = GridSearchCV(gradient_boosting, param_grid, cv=number_of_splits, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    return clf
