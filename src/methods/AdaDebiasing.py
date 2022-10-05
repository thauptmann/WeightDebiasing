from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def cv_bootstrap_prediction(df, number_of_splits, columns):
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    predictions = np.zeros(len(N))

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    for train_index, test_index in kf.split(N):
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        x_train = pd.concat([N_train, R])
        clf = train_tree(
            x_train[columns], x_train.label, x_train["weights"], number_of_splits
        )
        predictions[test_index] = clf.predict_proba(N_test[columns])[:, 1]
    return predictions, clf


def train_tree(X_train, y_train, weights, number_of_splits):
    param_grid = {"max_depth": [1, 2, 5], "min_samples_split": [2, 5]}
    decision_stump = GridSearchCV(
        DecisionTreeClassifier(), param_grid, n_jobs=-1, cv=number_of_splits
    )
    decision_stump = decision_stump.fit(X_train, y_train, sample_weight=weights)
    return decision_stump
