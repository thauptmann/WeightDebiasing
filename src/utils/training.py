import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.metrics import interpolate_roc, calculate_median_roc


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
