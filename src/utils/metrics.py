import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd


def interpolate_roc(y_test, y_predict, iteration):
    interpolation_points = 250
    median_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
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


def average_standardised_absolute_mean_distance(df, columns, weights):
    N = df[df['label'] == 1][columns]
    R = df[df['label'] == 0][columns]
    means_representative = np.mean(R, axis=0)
    weighted_means_non_representative = np.average(N, axis=0, weights=weights)
    variance_representative = np.var(R, axis=0)
    weighted_variance_non_representative = np.average((N-weighted_means_non_representative)**2, weights=weights,
                                                      axis=0)
    means_difference = means_representative-weighted_means_non_representative
    variances_difference = np.sqrt((variance_representative + weighted_variance_non_representative)/2)
    standardised_absolute_mean_distances = abs(means_difference / variances_difference)
    return standardised_absolute_mean_distances


def calculate_rbf_gamma(aggregate_set):
    all_distances = pdist(aggregate_set.values, 'euclid')
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma ** 2))


def weighted_maximum_mean_discrepancy(x, y, weights):
    min_weight = np.min(weights)
    tmp = np.round(weights / min_weight)
    new_x = pd.DataFrame(np.repeat(x.values, tmp, axis=0))
    new_x.columns = x.columns
    return maximum_mean_discrepancy(new_x, y)


def maximum_mean_discrepancy(x, y):
    gamma = calculate_rbf_gamma(pd.concat([x, y]))
    x_x_rbf_matrix = rbf_kernel(x, x, gamma)
    y_y_rbf_matrix = rbf_kernel(y, y, gamma)
    x_y_rbf_matrix = rbf_kernel(x, y, gamma)
    a = 1 / (len(x) * len(x))
    b = 2 / (len(x) * len(y))
    c = 1 / (len(y) * len(y))
    maximum_mean_discrepancy_value = (a * x_x_rbf_matrix.sum()) - (b * x_y_rbf_matrix.sum()) \
                               + (c * y_y_rbf_matrix.sum())
    return np.sqrt(maximum_mean_discrepancy_value)
