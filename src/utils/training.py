import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.metrics import interpolate_roc, calculate_median_roc


def train_stump(X_train, y_train, weights):
    decision_stump = DecisionTreeClassifier(max_depth=2)
    decision_stump = decision_stump.fit(X_train, y_train, sample_weight=weights)
    return decision_stump


def train_forest(X_train, y_train, weights):
    random_forest = DecisionTreeClassifier()
    random_forest = random_forest.fit(X_train, y_train, sample_weight=weights)
    return random_forest


def weighted_auc_prediction(data, columns, iteration, calculate_roc=False):
    auroc_scores = []
    rocs = []
    median_roc = None
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in kf.split(data[columns], data['label']):
        train, test = data.iloc[train], data.iloc[test]
        y_train = train['label']
        clf = train_forest(train[columns], y_train, train.weights)
        y_predict = clf.predict_proba(test[columns])[:, 1]
        y_test = test['label']
        auroc_scores.append(roc_auc_score(y_test, y_predict))
        if calculate_roc:
            rocs.append(interpolate_roc(y_test, y_predict, iteration))
    if calculate_roc:
        median_roc = calculate_median_roc(rocs)

    return np.mean(auroc_scores), median_roc


def cv_bootstrap_prediction(df, number_of_splits, columns):
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    preds = np.zeros(len(N))
    preds_r = np.zeros(len(R))
    bootstrap_iterations = 10

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, R_test = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap_predictions = []
        bootstrap_predictions_r = []
        for j in range(bootstrap_iterations):
            bootstrap = pd.concat([N_train.sample(n=n, replace=True),
                                   R_train.sample(n=n, replace=True)])
            clf = train_stump(bootstrap[columns], bootstrap.label, bootstrap.weights)
            bootstrap_predictions.append(clf.predict_proba(N_test[columns])[:, 1])
            bootstrap_predictions_r.append(clf.predict_proba(R_test[columns])[:, 1])
        preds[test_index] = np.mean(bootstrap_predictions, axis=0)
        preds_r[test_index_r] = np.mean(bootstrap_predictions_r, axis=0)
    return preds, preds_r, clf


def train_classifier(df, clf, columns):
    clf = clf.fit(df[columns], df.label)
    return clf.predict_proba(df[columns])[:, 1]
