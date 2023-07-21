import random
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_relative_bias,
    compute_test_metrics_mrs,
    train_pu_classifier,
    weighted_maximum_mean_discrepancy,
)


def mrs(
    N,
    R,
    columns,
    n_drop: int = 1,
    cv=5,
    class_weights="balanced",
    sampling="temperature",
    *args,
    **attributes
):
    """_summary_

    :param N: _description_
    :param R: _description_
    :param columns: _description_
    :param n_drop: _description_, defaults to 1
    :param cv: _description_, defaults to 5
    :param class_weights: _description_, defaults to "balanced"
    :param sampling: _description_, defaults to "temperature"
    :return: _description_
    """
    all_predictions = np.zeros(len(N))
    kf = KFold(n_splits=cv, shuffle=True)
    for train_index, test_index in kf.split(N):
        N_train, N_test = N.iloc[train_index], N.iloc[test_index]
        data = pd.concat([N_train, R])
        clf = train_pu_classifier(data[columns], data.label, class_weight=class_weights)
        predictions = clf.predict_proba(N_test[columns])[:, 1]
        all_predictions[test_index] = predictions

    drop_ids = np.argpartition(all_predictions, -n_drop)[-n_drop:]
    drop_index = N.index[drop_ids]
    return N.drop(N.index[drop_ids]), drop_index


def mrs_without_cv(
    N,
    R,
    columns,
    n_drop: int = 1,
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
    data = pd.concat([N, R])
    clf = train_pu_classifier(data[columns], data.label, class_weight=class_weights)
    predictions = clf.predict_proba(N[columns])[:, 1]
    drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]

    drop_index = N.index[drop_ids]
    return N.drop(N.index[drop_ids]), drop_index


def repeated_MRS(
    N,
    R,
    columns,
    delta=0.001,
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
        if (auc_difference + delta) <= best_difference:
            best_weights = weights.copy()
            mrs_iteration = (i + 1) * drop
            best_difference = auc_difference

        if (
            len(dropping_N) <= cv
            or ((best_difference <= delta) and early_stopping)
            or len(dropping_N) <= drop
        ):
            break

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
