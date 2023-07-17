import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

    for i in range(scaled_N.values.shape[1]):
        u_values = scaled_N.values[:, i]
        v_values = scaled_R.values[:, i]

        wasserstein_distance_value = wasserstein_distance(u_values, v_values, weights)
        wasserstein_distances.append(wasserstein_distance_value)

    scaled_N.loc[:, scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R.loc[:, scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

    sample_biases = compute_relative_bias(scaled_N, scaled_R, weights)

    return (
        weighted_mmd,
        sample_biases,
        wasserstein_distances,
    )


def compute_classification_metrics(N, R, columns, weights, label):
    y_true = R[label]
    # clf = train_classifier(N[columns], N[label], weights)
    clf = train_classifier_auroc(N[columns], N[label], weights)
    y_predictions = clf.predict_proba(R[columns])[:, 1]
    auroc_score = roc_auc_score(y_true, y_predictions)
    auprc = average_precision_score(y_true, y_predictions.round())

    return auroc_score, auprc


def train_classifier(X, y, weights):
    clf = RandomForestClassifier(n_jobs=-1)
    new_weights = weights * len(X)
    clf = clf.fit(X, y, sample_weight=new_weights)
    return clf


def compute_regression_metrics(N, R, columns, weights, label):
    regressor = train_regressor(N[columns], N[label], weights)
    y_prediction = regressor.predict(R[columns])
    mean_y = np.mean(R[label])

    mse = np.sqrt(mean_squared_error(R[label], y_prediction))
    nmse = mse / mean_y
    return nmse


def train_regressor(X, y, weights):
    new_weights = weights * len(X)
    clf = RandomForestRegressor()
    clf = clf.fit(X, y, sample_weight=new_weights)
    return clf


def compute_test_metrics_mrs(data, columns, calculate_roc=False, weights=None, cv=3):
    if weights is None:
        weights = np.ones(len(data)) / len(data)
    auroc_scores = []
    ifpr_list = []
    itpr_list = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True)
    for train_indices, test_indices in kf.split(data[columns], data["label"]):
        train, test = data.iloc[train_indices], data.iloc[test_indices]
        train_weights = weights[train_indices]
        clf = train_classifier_auroc(train[columns], train.label, weights=train_weights)
        y_predict = clf.predict_proba(test[columns])[:, 1]
        auroc = roc_auc_score(test.label, y_predict)
        auroc_scores.append(auroc)
        if calculate_roc:
            interpolated_fpr, interpolated_tpr = interpolate_roc(test.label, y_predict)
            ifpr_list.append(interpolated_fpr)
            itpr_list.append(interpolated_tpr)
    if calculate_roc:
        mean_ifpr_list, mean_itpr_list, std_tpr = calculate_mean_roc(
            ifpr_list, itpr_list
        )
        return np.mean(auroc_scores), mean_ifpr_list, mean_itpr_list, std_tpr
    else:
        return np.mean(auroc_scores)


def compute_test_metrics_ada_deboost(data, columns, weights):
    clf = train_classifier_auroc(data[columns], data.label, weights=weights)
    test_N = data[data["label"] == 1]
    y_predict_N = clf.predict_proba(test_N[columns])[:, 1]
    y_predict = clf.predict_proba(data[columns])[:, 1]
    auroc = roc_auc_score(data.label, y_predict)
    return y_predict_N, auroc


def train_pu_classifier(X_train, y_train, class_weight="balanced"):
    clf = RandomForestClassifier(
        class_weight=class_weight, n_estimators=25, n_jobs=-1, max_depth=25
    )
    return clf.fit(X_train, y_train)


def interpolate_roc(y_test, y_predict):
    interpolation_points = 250
    interpolated_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    interpolated_tpr = np.interp(interpolated_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0
    return interpolated_fpr, interpolated_tpr


def train_classifier_auroc(X_train, y_train, weights=None, speedup=True, cv=3):
    if weights is None:
        weights = np.ones(len(X_train)) / len(X_train)
    clf = DecisionTreeClassifier()
    path = clf.cost_complexity_pruning_path(X_train, y_train, sample_weight=weights)
    ccp_alphas = path.ccp_alphas
    ccp_alphas[ccp_alphas < 0] = 0
    ccp_alphas_unique = np.unique(ccp_alphas)

    if speedup:
        if len(ccp_alphas_unique) > 10:
            shortened_ccp_alphas_unique = ccp_alphas_unique[0::10]
            ccp_alphas_unique = np.append(
                ccp_alphas_unique[-10:], shortened_ccp_alphas_unique
            )
    param_grid = {"ccp_alpha": ccp_alphas_unique}
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    return grid.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )


def calculate_mean_rocs(rocs):
    rocs = np.array(rocs, dtype=object)
    mean_rocs = []
    for i in range(rocs.shape[1]):
        rocs_at_iteration = rocs[:, i]
        mean_fpr, mean_tpr, std_tpr = calculate_mean_roc(
            rocs_at_iteration[:, 0], rocs_at_iteration[:, 1]
        )
        removed_samples = rocs_at_iteration[0, 3]
        mean_rocs.append((mean_fpr, mean_tpr, std_tpr, removed_samples))
    return mean_rocs


def calculate_mean_roc(interpolated_fpr, interpolated_tpr):
    mean_fpr = np.mean(interpolated_fpr, axis=0)
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    std_tpr = np.std(interpolated_tpr, axis=0)
    return mean_fpr, mean_tpr, std_tpr
