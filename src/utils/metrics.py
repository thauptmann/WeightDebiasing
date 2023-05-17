import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    roc_curve,
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, ElasticNet
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier


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


def compute_relative_bias(N, R, weights):
    weighted_means = compute_weighted_means(N, weights)
    population_means = np.mean(R, axis=0)
    return (abs(weighted_means - population_means) / population_means) * 100


def compute_bias(N, R, weights):
    weighted_means = compute_weighted_means(N, weights)
    population_means = np.mean(R, axis=0)
    return abs(weighted_means - population_means)


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
    gamma, n, r, weights, n_n_rbf_matrix=None, r_r_rbf_matrix=None, n_r_rbf_matrix=None
):
    weights_r = np.ones(len(r)) / len(r)
    weights_n = weights / np.sum(weights)

    if n_n_rbf_matrix is None:
        n_n_rbf_matrix = rbf_kernel(n, n, gamma=gamma)
    weights_n_n = np.matmul(np.expand_dims(weights_n, 1), np.expand_dims(weights_n, 0))
    n_n_mean = (weights_n_n * n_n_rbf_matrix).sum()

    if r_r_rbf_matrix is None:
        r_r_rbf_matrix = rbf_kernel(r, r, gamma=gamma)
    weight_matrix_r_r = np.matmul(
        np.expand_dims(weights_r, 1), np.expand_dims(weights_r, 0)
    )
    r_r_mean = (weight_matrix_r_r * r_r_rbf_matrix).sum()

    if n_r_rbf_matrix is None:
        n_r_rbf_matrix = rbf_kernel(n, r, gamma=gamma)
    weight_matrix_n_r = np.matmul(
        np.expand_dims(weights_n, 1), np.expand_dims(weights_r, 0)
    )
    n_r_mean = (weight_matrix_n_r * n_r_rbf_matrix).sum()

    mmd = n_n_mean + r_r_mean - 2 * n_r_mean
    return np.sqrt(mmd)


def weighted_maximum_mean_discrepancy(
    x,
    y,
    weights,
    gamma=None,
    x_x_rbf_matrix=None,
    y_y_rbf_matrix=None,
    x_y_rbf_matrix=None,
):
    if gamma is None:
        gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    return compute_weighted_maximum_mean_discrepancy(
        gamma, x, y, weights, x_x_rbf_matrix, y_y_rbf_matrix, x_y_rbf_matrix
    )


def compute_metrics(scaled_N, scaled_R, weights, scaler, scale_columns, columns, gamma):
    wasserstein_distances = []
    scaled_N_dropped = scaled_N[columns].values
    scaled_R_dropped = scaled_R[columns].values

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

    for i in range(scaled_N.values.shape[1]):
        u_values = scaled_N.values[:, i]
        v_values = scaled_R.values[:, i]

        wasserstein_distance_value = wasserstein_distance(u_values, v_values, weights)
        wasserstein_distances.append(wasserstein_distance_value)

    scaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

    sample_biases = compute_relative_bias(scaled_N, scaled_R, weights)

    return (
        weighted_mmd,
        weighted_ssmd,
        sample_biases,
        wasserstein_distances,
    )


def compute_classification_metrics(N, R, columns, weights, label):
    clf = train_classifier(N[columns], N[label], weights)
    y_probabilities = clf.predict_proba(R[columns])[:, 1]

    auroc_score = roc_auc_score(R[label], y_probabilities)
    accuracy = accuracy_score(R[label], y_probabilities.round())
    precision = precision_score(R[label], y_probabilities.round())
    recall = recall_score(R[label], y_probabilities.round())

    return auroc_score, accuracy, precision, recall


def train_classifier(X, y, weights):
    # pool = Pool(X, y, weight=weights)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y, sample_weight=weights)
    return clf


def compute_regression_metrics(N, R, columns, weights, label):
    clf = train_regressor(N[columns], N[label], weights)
    y_prediction = clf.predict(R[columns])
    mse = mean_squared_error(R[label], y_prediction)
    return mse


def train_regressor(X, y, weights):
    train_pool = Pool(X, y, weight=weights)
    clf = CatBoostRegressor(verbose=0)
    clf.fit(train_pool)
    return clf


def compute_test_metrics_mrs(data, columns, cv=5, calculate_roc=False):
    auroc_scores = []
    ifpr_list = []
    itpr_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(data[columns], data["label"]):
        train, test = data.iloc[train_indices], data.iloc[test_indices]
        y_train = train["label"]
        clf = train_classifier_mrs(train[columns], y_train)
        y_predict = clf.predict_proba(test[columns])[:, 1]
        y_test = test["label"]
        auroc_scores.append(roc_auc_score(y_test, y_predict))
        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(y_test, y_predict)
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)
    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, std_tpr = calculate_mean_roc(
            ifpr_list, itpr_list
        )
        return np.mean(auroc_scores), mean_ifpr_list, mean_itpr_list, std_tpr
    else:
        return np.mean(auroc_scores)


def train_classifier_mrs(X_train, y_train):
    # clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf = DecisionTreeClassifier(class_weight="balanced")
    return clf.fit(X_train, y_train)


def train_classifier_test(X_train, y_train):
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced")
    return clf.fit(X_train, y_train)


def train_classifier_mrs_auroc(X_train, y_train, cv=5):
    clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    ccp_alphas_unique = np.unique(ccp_alphas)
    ccp_alphas_unique[ccp_alphas_unique < 0] = 0

    param_grid = {"ccp_alpha": ccp_alphas_unique[:-1]}
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def calculate_mean_roc(interpolated_fpr, interpolated_tpr):
    mean_fpr = np.mean(interpolated_fpr, axis=0)
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    std_tpr = np.std(interpolated_tpr, axis=0)
    return mean_fpr, mean_tpr, std_tpr


def interpolate_roc(y_test, y_predict):
    interpolation_points = 250
    interpolated_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    interpolated_tpr = np.interp(interpolated_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0
    return interpolated_fpr, interpolated_tpr
