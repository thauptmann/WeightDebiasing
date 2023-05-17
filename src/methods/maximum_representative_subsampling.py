import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from utils.metrics import (
    calculate_rbf_gamma,
    compute_bias,
    compute_test_metrics_mrs,
    train_classifier_mrs,
    train_classifier_test,
    weighted_maximum_mean_discrepancy,
)
from utils.visualization import mrs_progress_visualization, plot_class_ratio, plot_rocs


def temperature_sample(softmax: list, temperature: float, drop: int):
    EPSILON = 10e-16  # to avoid taking the log of zero
    softmax = (np.array(softmax)).astype("float64")
    softmax[softmax == 0] = EPSILON
    predictions = np.log(softmax) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(predictions), drop, replace=False, p=predictions)


def cv_bootstrap_prediction(N, R, number_of_splits, columns, cv):
    preds = np.zeros(len(N))
    preds_r = np.zeros(len(R))

    # kf = KFold(n_splits=cv, shuffle=True)
    # for split_n, split_r in zip(kf.split(N), kf.split(R)):
    #    train_index, test_index = split_n
    #    train_index_r, test_index_r = split_r
    # N_train, N_test = N.iloc[train_index], N.iloc[test_index]
    # R_train, R_test = R.iloc[train_index_r], R.iloc[test_index_r]

    data = pd.concat([N, R])
    clf = train_classifier_test(data[columns], data.label)
    predictions = clf.predict_proba(N[columns])[:, 1]
    predictions_r = clf.predict_proba(R[columns])[:, 1]
    return predictions, predictions_r


def mrs(
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

    predictions, preds_r = cv_bootstrap_prediction(N, R, number_of_splits, columns, cv)
    all_preds = np.concatenate([predictions, preds_r])
    all_true = np.concatenate([np.ones(len(predictions)), np.zeros(len(preds_r))])
    auroc = roc_auc_score(all_true, all_preds)
    if temperature_sampling:
        mapped_auc = abs(auroc - 0.5)
        temperature = -0.55 * mapped_auc + 0.3
    else:
        temperature = 1
    drop_ids = temperature_sample(predictions, temperature, n_drop)
    # drop_ids = np.argpartition(predictions, -n_drop)[-n_drop:]
    return N.drop(N.index[drop_ids]), N.index[drop_ids]


def repeated_MRS(
    N,
    R,
    columns,
    number_of_splits,
    delta=0.01,
    cv=5,
    early_stopping=False,
    mrs_function=mrs,
    ablation_study=False,
    use_bias_mean=True,
    temperature_sampling=True,
    bias_variable=None,
    *args,
    **attributes
):
    drop = attributes["drop"]
    save_path = attributes["save_path"]
    auc_list = []
    mean_rocs_list = []
    median_rocs_list = []
    relative_bias_list = []
    deleted_element_list = []
    mmd_list = []

    number_of_iterations = len(N) // drop
    number_of_splits = 5
    mrs_iteration = 0
    auroc_iteration = int(int(len(N) / drop) / 3.5) + 1
    dropping_N = N.copy()
    weights = np.ones(len(N))
    dropping_N = dropping_N.reset_index(drop=True)
    best_difference = np.inf

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
    auc, _, _ = compute_test_metrics_mrs(
        pd.concat([dropping_N, R]),
        columns,
        cv,
    )

    auc_list.append(auc)

    for i in tqdm(range(number_of_iterations)):
        dropping_N, drop_ids = mrs_function(
            dropping_N,
            R,
            columns,
            number_of_splits=number_of_splits,
            n_drop=drop,
            cv=cv,
            temperature_sampling=temperature_sampling,
        )
        weights[drop_ids] = 0

        if (i + 1) % auroc_iteration == 0:
            auc, mean_ifpr_list, mean_itpr_list, std_tpr = compute_test_metrics_mrs(
                pd.concat([dropping_N, R]),
                columns,
                cv,
            )
            deleted_element_list.append(i * drop)
        else:
            auc, _, _ = compute_test_metrics_mrs(
                pd.concat([dropping_N, R]),
                columns,
                cv,
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
            relative_bias = compute_bias(N[bias_variable], R[bias_variable], weights)
            relative_bias_list.append(relative_bias)
            if (i % 10) == 0:
                plot_class_ratio(
                    relative_bias_list,
                    0,
                    save_path / "Relative_Bias",
                    [mrs_iteration],
                    len(N),
                    drop,
                )

        auc_difference = abs(auc - 0.5)
        if (len(dropping_N) - drop) < cv:
            break
        if auc_difference < best_difference:
            best_weights = weights
            mrs_iteration = (i + 1) * drop
            best_difference = auc_difference

            if early_stopping:
                break

        if (i % 10) == 0:
            mrs_progress_visualization(
                [mmd_list],
                [auc_list],
                np.array(relative_bias_list),
                [mrs_iteration],
                drop,
                len(N),
                save_path,
            )

    best_weights = best_weights.astype(np.float64)

    plot_rocs(
        mean_ifpr_list,
        mean_itpr_list,
        deleted_element_list,
        std_tpr,
        save_path / "mean_rocs",
    )

    if ablation_study:
        return None
    else:
        return best_weights / best_weights.sum()


def mrs_without_cv(
    N,
    R,
    columns,
    number_of_splits=5,
    n_drop: int = 5,
    cv=5,
    temperature_sampling=True,
):
    EPSILON = 10e-16  # to avoid dividing by zero
    bootstrap_iterations = 25
    bootstrap_predictions_n = np.zeros(len(N))
    bootstrap_predictions_r = np.zeros(len(R))
    counter_n = np.zeros(len(N))
    counter_r = np.zeros(len(R))

    n = min(len(R), len(N))
    for _ in range(bootstrap_iterations):
        n_sample = N.sample(n=n, replace=True)
        N_test = N.drop(n_sample.index)
        r_sample = R.sample(n=n, replace=True)
        R_test = R.drop(r_sample.index)
        locations_not_in_bootstrap_n = list(
            set([N.index.get_loc(index) for index in N_test.index])
        )
        locations_not_in_bootstrap_r = list(
            set([R.index.get_loc(index) for index in R_test.index])
        )

        bootstrap = pd.concat([n_sample, r_sample])
        clf = train_classifier_mrs(bootstrap[columns], bootstrap.label, 5)
        proba_n = clf.predict_proba(N_test[columns])[:, 1]
        proba_r = clf.predict_proba(R_test[columns])[:, 1]
        bootstrap_single_n = np.zeros(len(N))
        bootstrap_single_n[list(locations_not_in_bootstrap_n)] = proba_n
        counter_n[list(locations_not_in_bootstrap_n)] += 1
        bootstrap_predictions_n += bootstrap_single_n

        bootstrap_single_r = np.zeros(len(R))
        bootstrap_single_r[list(locations_not_in_bootstrap_r)] = proba_r
        counter_r[list(locations_not_in_bootstrap_r)] += 1
        bootstrap_predictions_r += bootstrap_single_r

    counter_n = [EPSILON if x == 0 else x for x in counter_n]
    counter_r = [EPSILON if x == 0 else x for x in counter_r]
    preds_n = bootstrap_predictions_n / counter_n
    preds_r = bootstrap_predictions_r / counter_r

    all_preds = np.concatenate([preds_n, preds_r])
    all_true = np.concatenate([np.ones(len(preds_n)), np.zeros(len(preds_r))])
    auc = roc_auc_score(all_true, all_preds)
    mapped_auc = abs(auc - 0.5)
    temperature = -0.55 * mapped_auc + 0.3
    drop_ids = temperature_sample(preds_n, temperature, n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]


def random_drops(
    N,
    R,
    columns,
    number_of_splits=5,
    n_drop: int = 5,
    cv=5,
    temperature_sampling=True,
):
    drop_ids = random.sample(range(0, len(N)), n_drop)
    return N.drop(N.index[drop_ids]), N.index[drop_ids]
