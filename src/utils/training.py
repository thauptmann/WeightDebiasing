import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.metrics import interpolate_roc, calculate_median_roc


def train_tree(X_train, y_train, weights=None):
    if weights is not None:
        param_grid = {'max_depth': [1, 2, 5], 'min_samples_split': [2, 5]}
        decision_stump = GridSearchCV(
            DecisionTreeClassifier(), param_grid, n_jobs=-1)
        decision_stump = decision_stump.fit(
            X_train, y_train, sample_weight=weights)
    else:
        forest = RandomForestClassifier(max_depth=5, n_estimators=50)
        decision_stump = forest.fit(X_train, y_train)
    return decision_stump


def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression()
    logistic_regression = logistic_regression.fit(X_train, y_train)
    return logistic_regression


def weighted_auc_prediction(data, columns, iteration, calculate_roc=False):
    auroc_scores = []
    rocs = []
    median_roc = None
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in kf.split(data[columns], data['label']):
        train, test = data.iloc[train], data.iloc[test]
        y_train = train['label']
        clf = train_tree(train[columns], y_train, train.weights)
        y_predict = clf.predict_proba(test[columns])[:, 1]
        y_test = test['label']
        auroc_scores.append(roc_auc_score(y_test, y_predict))
        if calculate_roc:
            rocs.append(interpolate_roc(y_test, y_predict, iteration))
    if calculate_roc:
        median_roc = calculate_median_roc(rocs)

    return np.mean(auroc_scores), median_roc


def cv_bootstrap_prediction(df, number_of_splits, columns, use_weights=True):
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    predictions = np.zeros(len(N))
    bootstrap_iterations = 10

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, _ = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap_predictions = []
        for _ in range(bootstrap_iterations):
            bootstrap = pd.concat([N_train.sample(n=n, replace=True),
                                   R_train.sample(n=n, replace=True)])
            if not use_weights:
                clf = train_tree(bootstrap[columns], bootstrap.label)
            else:
                clf = train_tree(bootstrap[columns], bootstrap.label,
                                 bootstrap['weights'])
            bootstrap_predictions.append(
                clf.predict_proba(N_test[columns])[:, 1])
        predictions[test_index] = np.mean(bootstrap_predictions, axis=0)
    return predictions, clf


def neural_network_prediction(df, columns, number_of_splits, *args, **attributes):
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    predictions = np.zeros(len(N))
    bootstrap_iterations = 100

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, _ = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap_predictions = []
        for _ in tqdm(range(bootstrap_iterations)):
            bootstrap = pd.concat([N_train.sample(n=n, replace=True),
                                   R_train.sample(n=n, replace=True)])
            clf = train_neural_network(bootstrap[columns], bootstrap.label)
            bootstrap_predictions.append(
                clf.predict_proba(N_test[columns].values)[:, 1])
        predictions[test_index] = np.mean(bootstrap_predictions, axis=0)
    return predictions, clf


def train_neural_network(X_train, y_train):
    features = np.shape(X_train)[1]
    nn = MLPClassifier(hidden_layer_sizes=[int(features/2)], max_iter=1000,
                       learning_rate_init=0.01,
                       batch_size=64,
                       early_stopping=True, learning_rate='adaptive')
    nn = nn.fit(X_train.values, y_train.values)
    return nn


def logistic_regression_prediction(df, number_of_splits, columns, *args, **attributes):
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    predictions = np.zeros(len(N))
    bootstrap_iterations = 10

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, _ = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap_predictions = []
        for _ in range(bootstrap_iterations):
            bootstrap = pd.concat([N_train.sample(n=n, replace=True),
                                   R_train.sample(n=n, replace=True)])
            clf = train_logistic_regression(
                bootstrap[columns], bootstrap.label)
            bootstrap_predictions.append(
                clf.predict_proba(N_test[columns])[:, 1])
        predictions[test_index] = np.mean(bootstrap_predictions, axis=0)
    return predictions, clf
