import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import ot
import torch


def strictly_standardized_mean_difference(N, R, weights=None):
    if weights is None:
        weights = np.ones(len(N))
    N = N.numpy()
    R = R.numpy()
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
    weights = weights / sum(weights)
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
        gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    return compute_weighted_maximum_mean_discrepancy(gamma, x, y, weights)


def compute_metrics(scaled_N, scaled_R, weights, scaler, scale_columns, gamma):
    wasserstein_distances = []
    if isinstance(weights, (np.ndarray)):
        weights = torch.DoubleTensor(weights)
    scaled_N_dropped = torch.DoubleTensor(
        scaled_N.drop(["pi", "label"], axis="columns").values
    )
    scaled_R_dropped = torch.DoubleTensor(
        scaled_R.drop(["pi", "label"], axis="columns").values
    )

    weighted_mmd = maximum_mean_discrepancy_weighted(
        scaled_N_dropped,
        scaled_R_dropped,
        weights,
        gamma,
    )
    weighted_ssmd = strictly_standardized_mean_difference(
        scaled_N_dropped,
        scaled_R_dropped,
        weights,
    )

    data_set_wasserstein = WassersteinMetric(
        scaled_N_dropped, scaled_R_dropped, weights
    )

    for i in range(scaled_N_dropped.shape[1]):
        u_values = scaled_N_dropped[:, i]
        v_values = scaled_R_dropped[:, i]

        wasserstein_distance_value = wasserstein_distance(u_values, v_values, weights)
        wasserstein_distances.append(wasserstein_distance_value)

    scaled_N_dropped = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R_dropped = scaler.inverse_transform(scaled_R[scale_columns])
    weighted_means = compute_weighted_means(scaled_N_dropped, weights)

    sample_means = np.mean(scaled_R_dropped, axis=0)
    sample_biases = compute_relative_bias(weighted_means, sample_means)

    return (
        weighted_mmd,
        weighted_ssmd,
        sample_biases,
        wasserstein_distances,
        data_set_wasserstein,
    )


def WassersteinMetric(N, R, weights, method="emd"):
    uniform_weights = (torch.ones(len(R), dtype=torch.float64) / len(R)).to(R)
    M = ot.dist(N, R)
    if method == "emd":
        return ot.emd2(weights, uniform_weights, M)
    else:
        return ot.sinkhorn2(weights, uniform_weights, M, reg=1)
