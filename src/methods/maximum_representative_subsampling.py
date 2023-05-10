import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from utils.metrics import compute_test_metrics_mrs, train_classifier_mrs
from tqdm import tqdm


def temperature_sample(softmax: list, temperature: float, drop: int):
    EPSILON = 10e-16  # to avoid taking the log of zero
    softmax = (np.array(softmax)).astype("float64")
    softmax[softmax == 0] = EPSILON
    preds = np.log(softmax) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    count = 0
    while np.isnan(preds).any() and count < 100:
        preds = exp_preds / np.sum(exp_preds)
        count += 1

    if count == 100:
        return []

    if len(preds[preds != 0]) < drop:
        drop = preds[preds != 0]

    return np.random.choice(len(preds), drop, replace=False, p=preds)


def cv_bootstrap_prediction(N, R, number_of_splits, columns, cv):
    kf = KFold(n_splits=cv, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, R_test = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap = pd.concat(
            [N_train.sample(n=n, replace=True), R_train.sample(n=n, replace=True)]
        )
        clf = train_classifier_mrs(
            bootstrap[columns], bootstrap.label, number_of_splits
        )
        preds = clf.predict_proba(N_test[columns])[:, 1]
        preds_r = clf.predict_proba(R_test[columns])[:, 1]
    return preds, preds_r


def MRS(
    N, R, columns, number_of_splits=5, n_drop: int = 5, cv=5, temperature_sampling=True
):
    """
    MRS Algorithm

    Input:
        * N: dataset that is assumed to not be representative.
        * R: dataset that is known to be representative.
        * temperature: temperature value for probabilistic sampling procedure.
        * drop: number of instances to drop per iteration (small values result in long runtimes).
        * number_of_splits: splits per iteration.

    Output:
        * N/Drop: N without the dropped elements
    """

    preds, preds_r = cv_bootstrap_prediction(N, R, number_of_splits, columns, cv)
    all_preds = np.concatenate([preds, preds_r])
    all_true = np.concatenate([np.ones(len(preds)), np.zeros(len(preds_r))])
    auc = roc_auc_score(all_true, all_preds)

    if temperature_sampling:
        mapped_auc = abs(auc - 0.5)
        temperature = -0.55 * mapped_auc + 0.3
    else:
        temperature = 1
    drop_ids = temperature_sample(preds, temperature, n_drop)

    return N.drop(N.index[drop_ids]), drop_ids


def repeated_MRS(N, R, columns, number_of_splits, *args, **attributes):
    drop = attributes["drop"]
    delta = 0.01
    temperature_sampling = (True,)
    cv = 5
    weights = np.ones(len(N))
    best_weights = weights[:]
    number_of_iterations = len(N) // drop
    number_of_splits = 5

    best_auc_difference = np.inf

    for i in tqdm(range(number_of_iterations)):
        N, drop_ids = MRS(
            N,
            R,
            columns,
            number_of_splits=number_of_splits,
            n_drop=drop,
            cv=cv,
            temperature_sampling=temperature_sampling,
        )
        weights[drop_ids] = 0

        auc, _, _ = compute_test_metrics_mrs(
            pd.concat([N, R]),
            columns,
            drop,
            i,
            cv,
        )

        if np.abs(auc - 0.5) <= delta:
            break

        elif abs(auc - 0.5) < best_auc_difference:
            best_auc_difference = abs(auc - 0.5)
            best_weights = weights

    best_weights = best_weights.astype(np.float64)
    return best_weights / best_weights.sum()
