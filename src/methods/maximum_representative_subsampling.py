import random
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    train_pu_classifier,
    weighted_maximum_mean_discrepancy,
)


def temperature_sample(softmax: list, temperature: float, drop: int):
    EPSILON = 10e-16  # to avoid taking the log of zero
    softmax = (np.array(softmax)).astype("float64")
    softmax[softmax == 0] = EPSILON
    predictions = np.log(softmax) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(predictions), drop, replace=False, p=predictions)


def pu_prediction(N, R, columns, class_weights):
    data = pd.concat([N, R])
    clf = train_pu_classifier(data[columns], data.label, class_weights)
    predictions = clf.predict_proba(N[columns])[:, 1]
    predictions_r = clf.predict_proba(R[columns])[:, 1]
    return predictions, predictions_r


def mrs(
    N, R, columns, n_drop: int = 5, cv=5, class_weights="balanced", *args, **attributes
):
    all_predictions = np.zeros(len(N))
    all_predictions_r = np.zeros(len(R))

    kf = KFold(n_splits=cv, shuffle=True)
    for split_n, split_r in zip(kf.split(N), kf.split(R)):
        train_index, test_index = split_n
        train_index_r, test_index_r = split_r
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        R_train, R_test = R.iloc[train_index_r], R.iloc[test_index_r]

        data = pd.concat([N_train, R_train])
        clf = train_pu_classifier(data[columns], data.label, class_weight=class_weights)
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        predictions_r = clf.predict_proba(R_test[columns])[:, 1]

        all_predictions[test_index] = predictions
        all_predictions_r[test_index_r] = predictions_r

    all_preds = np.concatenate([predictions, predictions_r])
    all_true = np.concatenate([np.ones(len(predictions)), np.zeros(len(predictions_r))])
    auc = roc_auc_score(all_true, all_preds)

    temperature = calculate_temperature(auc)
    drop_ids = temperature_sample(all_predictions, temperature, n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 5,
    sampling="temperature",
    class_weights="balanced",
    *args,
    **attributes
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

    predictions, preds_r = pu_prediction(N, R, columns, class_weights)
    all_preds = np.concatenate([predictions, preds_r])
    all_true = np.concatenate([np.ones(len(predictions)), np.zeros(len(preds_r))])
    auroc = roc_auc_score(all_true, all_preds)
    if sampling == "temperature" or sampling == "sampling":
        temperature = calculate_temperature(auroc) if sampling == "temperature" else 1
        drop_ids = temperature_sample(predictions, temperature, n_drop)
    elif sampling == "max":
        drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]

    return N.drop(N.index[drop_ids]), N.index[drop_ids]


def repeated_MRS(
    N,
    R,
    columns,
    delta=0.005,
    early_stopping=False,
    mrs_function=mrs,
    return_metrics=False,
    use_bias_mean=True,
    sampling="max",
    bias_variable=None,
    cv=5,
    class_weights="balanced",
    drop=1,
    *args,
    **attributes
):
    auc_list = []
    relative_bias_list = []
    mmd_list = []
    roc_list = []

    number_of_iterations = len(N) // drop
    mrs_iteration = 0
    roc_iteration = (len(N) // drop // 3.5) + 1
    dropping_N = N.copy()
    weights = np.ones(len(N))
    dropping_N = dropping_N.reset_index(drop=True)
    best_difference = np.inf

    # Compute and save mmd inputs to save time
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
    x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
    y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)

    # Start values
    mmd_list.append(
        weighted_maximum_mean_discrepancy(
            N[columns],
            R[columns],
            weights,
            gamma=gamma,
            x_x_rbf_matrix=x_x_rbf_matrix,
            x_y_rbf_matrix=x_y_rbf_matrix,
            y_y_rbf_matrix=y_y_rbf_matrix,
        )
    )
    auc, mean_ifpr_list, mean_itpr_list, std_tpr = compute_test_metrics_mrs(
        pd.concat([dropping_N, R]), columns, calculate_roc=True
    )
    roc_list.append([mean_ifpr_list, mean_itpr_list, std_tpr, 0])

    if use_bias_mean and bias_variable is not None:
        relative_bias = compute_relative_bias(
            N[bias_variable], R[bias_variable], weights
        )
        relative_bias_list.append(relative_bias)

    auc_list.append(auc)

    for i in trange(number_of_iterations):
        dropping_N, drop_ids = mrs_function(
            N=dropping_N,
            R=R,
            columns=columns,
            n_drop=drop,
            sampling=sampling,
            class_weights=class_weights,
            cv=cv,
        )
        weights[drop_ids] = 0

        if (i + 1) % roc_iteration == 0:
            auc, mean_ifpr_list, mean_itpr_list, std_tpr = compute_test_metrics_mrs(
                pd.concat([dropping_N, R]), columns, calculate_roc=True
            )
            roc_list.append([mean_ifpr_list, mean_itpr_list, std_tpr, i * drop])
        else:
            auc = compute_test_metrics_mrs(
                pd.concat([dropping_N, R]),
                columns,
            )

        auc_list.append(auc)

        mmd_list.append(
            weighted_maximum_mean_discrepancy(
                N[columns],
                R[columns],
                weights,
                gamma=gamma,
                x_x_rbf_matrix=x_x_rbf_matrix,
                x_y_rbf_matrix=x_y_rbf_matrix,
                y_y_rbf_matrix=y_y_rbf_matrix,
            )
        )

        if use_bias_mean and bias_variable is not None:
            relative_bias = compute_relative_bias(
                N[bias_variable], R[bias_variable], weights
            )
            relative_bias_list.append(relative_bias)

        auc_difference = abs(auc - 0.5)
        if (len(dropping_N) - drop) < cv or (
            (best_difference < delta) and early_stopping
        ):
            break

        if (auc_difference + delta) < best_difference:
            best_weights = weights.copy()
            mrs_iteration = (i + 1) * drop
            best_difference = auc_difference

    best_weights = best_weights.astype(np.float64)

    if return_metrics:
        return auc_list, mmd_list, relative_bias_list, mrs_iteration, roc_list
    else:
        return best_weights / best_weights.sum()


def calculate_temperature(auc):
    mapped_auc = abs(auc - 0.5)
    temperature = -0.55 * mapped_auc + 0.3
    return temperature


def random_drops(N, n_drop: int = 5, *args, **attributes):
    drop_ids = random.sample(range(0, len(N)), n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]
