import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.tree import DecisionTreeClassifier
import torch
import pandas as pd


def strictly_standardized_mean_difference(N, R, weights=None):
    if weights is None:
        weights = np.ones(len(N))
    means_representative = np.mean(R, axis=0)
    weighted_means_non_representative = np.average(N, axis=0, weights=weights)
    variance_representative = np.var(R, axis=0)
    weighted_variance_non_representative = np.average(
        (N - weighted_means_non_representative) ** 2, weights=weights, axis=0
    )
    means_difference = means_representative - weighted_means_non_representative
    middle_variance = np.sqrt(
        (variance_representative + weighted_variance_non_representative)
    )
    standardised_absolute_mean_distances = abs(means_difference / middle_variance)
    return standardised_absolute_mean_distances


def compute_weighted_means(N, weights):
    return np.average(N, weights=weights, axis=0)


def compute_relative_bias(weighted_means, population_means):
    return (abs(weighted_means - population_means) / population_means) * 100


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))


def maximum_mean_discrepancy(x, y):
    gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    return compute_maximum_mean_discrepancy(gamma, x, y)


def compute_maximum_mean_discrepancy(gamma, x, y):
    x_x_rbf_matrix = rbf_kernel(x, x, gamma)
    x_x_mean = x_x_rbf_matrix.mean()

    y_y_rbf_matrix = rbf_kernel(y, y, gamma)
    y_y_mean = y_y_rbf_matrix.mean()

    x_y_rbf_matrix = rbf_kernel(x, y, gamma)
    x_y_mean = x_y_rbf_matrix.mean()

    maximum_mean_discrepancy_value = x_x_mean + y_y_mean - 2 * x_y_mean
    return np.sqrt(maximum_mean_discrepancy_value)


def scale_df(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def compute_weighted_maximum_mean_discrepancy(
    gamma, x, y, weights, x_x_rbf_matrix=None, y_y_rbf_matrix=None, x_y_rbf_matrix=None
):
    weights_y = np.ones(len(y)) / len(y)
    if x_x_rbf_matrix is None:
        x_x_rbf_matrix = rbf_kernel(x, x, gamma=gamma)
    weights_x_x = np.matmul(np.expand_dims(weights, 1), np.expand_dims(weights, 0))
    x_x_mean = (weights_x_x * x_x_rbf_matrix).sum()

    if y_y_rbf_matrix is None:
        y_y_rbf_matrix = rbf_kernel(y, y, gamma=gamma)
    weight_matrix_y_y = np.matmul(
        np.expand_dims(weights_y, 1), np.expand_dims(weights_y, 0)
    )
    y_y_mean = (weight_matrix_y_y * y_y_rbf_matrix).sum()

    if x_y_rbf_matrix is None:
        x_y_rbf_matrix = rbf_kernel(x, y, gamma=gamma)
    weight_matrix_x_y = np.matmul(
        np.expand_dims(weights, 1), np.expand_dims(weights_y, 0)
    )
    x_y_mean = (weight_matrix_x_y * x_y_rbf_matrix).sum()

    maximum_mean_discrepancy_value = x_x_mean + y_y_mean - 2 * x_y_mean
    return np.sqrt(maximum_mean_discrepancy_value)


def weighted_maximum_mean_discrepancy(
    x,
    y,
    weights,
    gamma=None,
    x_x_rbf_matrix=None,
    y_y_rbf_matrix=None,
    x_y_rbf_matrix=None,
):
    weights = weights / sum(weights)
    if gamma is None:
        gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    return compute_weighted_maximum_mean_discrepancy(
        gamma, x, y, weights, x_x_rbf_matrix, y_y_rbf_matrix, x_y_rbf_matrix
    )


def compute_metrics(scaled_N, scaled_R, weights, scaler, scale_columns, columns, gamma):
    wasserstein_distances = []
    if isinstance(weights, (np.ndarray)):
        weights = torch.DoubleTensor(weights)
    scaled_N_dropped = torch.DoubleTensor(scaled_N[columns].values)
    scaled_R_dropped = torch.DoubleTensor(scaled_R[columns].values)

    weighted_mmd = weighted_maximum_mean_discrepancy(
        scaled_N_dropped,
        scaled_R_dropped,
        weights,
        gamma,
    )
    weighted_ssmd = strictly_standardized_mean_difference(
        scaled_N,
        scaled_R,
        weights,
    )

    weighted_ssmd_dataset = np.mean(
        strictly_standardized_mean_difference(
            scaled_N_dropped.numpy(),
            scaled_R_dropped.numpy(),
            weights,
        )
    )

    for i in range(scaled_N.values.shape[1]):
        u_values = scaled_N.values[:, i]
        v_values = scaled_R.values[:, i]

        wasserstein_distance_value = wasserstein_distance(u_values, v_values, weights)
        wasserstein_distances.append(wasserstein_distance_value)

    scaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])
    weighted_means = compute_weighted_means(scaled_N, weights)

    sample_means = np.mean(scaled_R, axis=0)
    sample_biases = compute_relative_bias(weighted_means, sample_means)

    return (
        weighted_mmd,
        weighted_ssmd,
        sample_biases,
        wasserstein_distances,
        weighted_ssmd_dataset,
    )


def auc_prediction(N, R, columns, weights, cv=5):
    data = pd.concat([N, R])
    representative_weights = np.ones(len(R)) * (len(R) / len(data))
    weights = np.concatenate([weights, representative_weights])
    clf = grid_search(data[columns], data.label, weights, cv)
    y_predict = clf.predict_proba(data[columns])[:, 1]
    auroc_score = roc_auc_score(data.label, y_predict)

    return auroc_score


def grid_search(X_train, y_train, weights, cv=5):
    clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(X_train, y_train, sample_weight=weights)
    ccp_alphas = path.ccp_alphas
    ccp_alphas[ccp_alphas < 0] = 0
    ccp_alphas = list(set(ccp_alphas[:-1]))
    param_grid = {"ccp_alpha": ccp_alphas}
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )
    return grid.best_estimator_
