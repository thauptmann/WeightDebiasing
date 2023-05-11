import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_test_metrics_mrs,
    train_classifier_mrs,
    weighted_maximum_mean_discrepancy,
)
from utils.visualization import mrs_progress_visualization


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
    preds = np.zeros(len(N))
    preds_r = np.zeros(len(R))
    bootstrap_iterations = 10

    kf = KFold(n_splits=cv, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, R_test = R.iloc[train_index_r], R.iloc[test_index_r]
        n = min(len(R_train), len(N_train))
        bootstrap_predictions = []
        bootstrap_predictions_r = []
        for _ in range(bootstrap_iterations):
            bootstrap = pd.concat(
                [N_train.sample(n=n, replace=True), R_train.sample(n=n, replace=True)]
            )
            clf = train_classifier_mrs(
                bootstrap[columns], bootstrap.label, number_of_splits
            )
            bootstrap_predictions.append(clf.predict_proba(N_test[columns])[:, 1])
            bootstrap_predictions_r.append(clf.predict_proba(R_test[columns])[:, 1])
        preds[test_index] = np.mean(bootstrap_predictions, axis=0)
        preds_r[test_index_r] = np.mean(bootstrap_predictions_r, axis=0)
    return preds, preds_r


def MRS(
    N,
    R,
    columns,
    number_of_splits=5,
    n_drop: int = 5,
    cv=5,
    temperature_sampling=True,
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


def repeated_MRS(
    N, R, columns, number_of_splits, delta=0.01, cv=5, *args, **attributes
):
    drop = attributes["drop"]
    save_path = attributes["save_path"]
    auc_list = []
    mean_rocs = []
    median_rocs = []
    ratio = []
    mmd_list = []
    temperature_sampling = (True,)
    weights = np.ones(len(N))
    best_weights = weights[:]
    number_of_iterations = len(N) // drop
    number_of_splits = 5
    dropping_N = N.copy()

    best_auc_difference = np.inf

    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
    x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
    y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)

    for i in tqdm(range(number_of_iterations)):
        dropping_N, drop_ids = MRS(
            dropping_N,
            R,
            columns,
            number_of_splits=number_of_splits,
            n_drop=drop,
            cv=cv,
            temperature_sampling=temperature_sampling,
        )
        weights[drop_ids] = 0

        auc, _, _ = compute_test_metrics_mrs(
            pd.concat([dropping_N, R]),
            columns,
            drop,
            i,
            cv,
        )

        auc_list.append(auc)
        mmd_list.append(
            weighted_maximum_mean_discrepancy(
                N,
                R,
                weights,
                gamma=gamma,
                x_x_rbf_matrix=x_x_rbf_matrix,
                x_y_rbf_matrix=x_y_rbf_matrix,
                y_y_rbf_matrix=y_y_rbf_matrix,
            )
        )
        mrs_progress_visualization([mmd_list], [auc_list], drop, number_of_iterations, save_path)

        if np.abs(auc - 0.5) <= delta:
            break

        elif abs(auc - 0.5) < best_auc_difference:
            best_auc_difference = abs(auc - 0.5)
            best_weights = weights

    best_weights = best_weights.astype(np.float64)
    return best_weights / best_weights.sum()
