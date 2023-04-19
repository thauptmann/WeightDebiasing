import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import rbf_kernel
from utils.metrics import calculate_rbf_gamma, weighted_maximum_mean_discrepancy
from scipy.special import xlogy

number_of_iterations = 100


def ada_debiasing_weighting(N, R, columns, number_of_splits, *args, **kwargs):
    best_mmd = np.inf
    best_weights = None
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    x_x_rbf_matrix = rbf_kernel(N[columns], N[columns], gamma=gamma)
    x_y_rbf_matrix = rbf_kernel(N[columns], R[columns], gamma=gamma)
    y_y_rbf_matrix = rbf_kernel(R[columns], R[columns], gamma=gamma)

    max_depths = [1, 3, 5, 7, 10]

    for max_depth in max_depths:
        weights = ada_debiasing(
            N,
            R,
            columns,
            max_depth,
        )
        mmd = weighted_maximum_mean_discrepancy(
            N[columns],
            R[columns],
            weights,
            gamma,
            x_x_rbf_matrix,
            y_y_rbf_matrix,
            x_y_rbf_matrix,
        )

        if mmd < best_mmd:
            best_mmd = mmd
            best_weights = weights

    return best_weights


def ada_debiasing(
    N,
    R,
    columns,
    max_depth,
):
    weights_N = np.ones(len(N)) / len(N)
    weights_R = np.ones(len(R)) / len(R)
    concat_data = pd.concat([N, R])
    x_label = concat_data.label
    x_train = concat_data[columns]
    y_codes = np.array([-1.0, 1.0])
    learning_rate = 1
    epsilon = np.finfo(weights_N.dtype).eps

    for _ in range(number_of_iterations):
        predictions = train_weighted_tree(
            x_train,
            x_label,
            np.concatenate([weights_N, weights_R]),
            max_depth,
        )

        predictions_N = predictions[: len(N), 1]
        predictions_N = np.clip(predictions_N, a_min=epsilon, a_max=None)

        y_coding = y_codes.take(predictions_N < 0.5)
        estimator_weight = -0.5 * learning_rate * xlogy(y_coding, predictions_N)
        weight_modificator = np.exp(estimator_weight)

        new_weights_N = weights_N * weight_modificator
        new_weights_N = new_weights_N / sum(new_weights_N)

        weights_N = new_weights_N

        learning_rate *= 0.95

    return weights_N


def train_weighted_tree(x_train, x_label, weights, max_depth):
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree = decision_tree.fit(x_train, x_label, sample_weight=weights)
    return decision_tree.predict_proba(x_train)
