import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


def interpolate_roc(y_test, y_predict, iteration):
    interpolation_points = 250
    median_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    interp_tpr = np.interp(median_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return median_fpr, interp_tpr, [iteration] * interpolation_points


def calculate_median_roc(rocs):
    rocs = np.array(rocs)
    median_fpr = np.median(rocs[:, 0], axis=0)
    median_tpr = np.median(rocs[:, 1], axis=0)
    std_tpr = np.std(rocs[:, 1], axis=0)
    removed_samples = rocs[0, 2]
    return median_fpr, median_tpr, std_tpr, removed_samples


def calculate_median_rocs(rocs):
    rocs = np.array(rocs)
    median_rocs = []
    for i in range(rocs.shape[1]):
        rocs_at_iteration = rocs[:, i]
        median_fpr = np.median(rocs_at_iteration[:, 0], axis=0)
        median_tpr = np.median(rocs_at_iteration[:, 1], axis=0)
        std_tpr = np.std(rocs_at_iteration[:, 1], axis=0)
        removed_samples = rocs_at_iteration[0, 3]
        median_rocs.append((median_fpr, median_tpr, std_tpr, removed_samples))
    return median_rocs


def average_standardised_absolute_mean_distance(N, R, columns, weights=None):
    N = N[columns]
    R = R[columns]

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
        (variance_representative + weighted_variance_non_representative) / 2
    )
    standardised_absolute_mean_distances = abs(means_difference / middle_variance)
    return standardised_absolute_mean_distances


def compute_relative_bias(N, R, weights):
    eps = 1e-20
    weights = weights / sum(weights)
    population_means = np.mean(R.values, axis=0) + eps
    weighted_means = np.average(N.values, weights=weights, axis=0) + eps
    return ((weighted_means - population_means) / population_means) * 100


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
    scaled = df.copy(deep=True)
    scaled[columns] = scaler.fit_transform(df[columns])
    return scaled, scaler


def compute_weighted_maximum_mean_discrepancy(gamma, x, y, weights):
    weights_y = np.ones(len(y)) / len(y)
    x_x_rbf_matrix = np.matmul(
        np.expand_dims(weights, 1), np.expand_dims(weights, 0)
    ) * rbf_kernel(x, x, gamma=gamma)
    x_x_mean = x_x_rbf_matrix.sum()

    y_y_rbf_matrix = rbf_kernel(y, y, gamma=gamma)
    y_y_mean = y_y_rbf_matrix.mean()
    weight_matrix = np.matmul(np.expand_dims(weights, 1), np.expand_dims(weights_y, 0))
    x_y_rbf_matrix = weight_matrix * rbf_kernel(x, y, gamma=gamma)
    x_y_mean = x_y_rbf_matrix.sum()

    maximum_mean_discrepancy_value = x_x_mean + y_y_mean - 2 * x_y_mean
    return np.sqrt(maximum_mean_discrepancy_value)


def maximum_mean_discrepancy_weighted(x, y, weights, gamma=None):
    weights = weights / sum(weights)
    if gamma is None:
        gamma = calculate_rbf_gamma_weighted(np.append(x, y, axis=0))
    return compute_weighted_maximum_mean_discrepancy(gamma, x, y, weights)


def calculate_rbf_gamma_weighted(aggregate_set):
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))


def compute_ratio(bias_values, weights):
    weights = np.squeeze(weights / np.sum(weights))
    one_indices = np.argwhere(bias_values == 1)
    zero_indices = np.argwhere(bias_values == 0)
    positive = np.sum(weights[one_indices])
    negative = np.sum(weights[zero_indices])
    return positive / negative
