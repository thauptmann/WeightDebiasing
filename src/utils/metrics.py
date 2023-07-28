import numpy as np

from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


def compute_weighted_means(N, weights):
    """Compute the weighted mean

    :param N: Non-representative data set
    :param weights: Sample weights
    :return: Weighted mean
    """
    return np.average(N, weights=weights, axis=0)


def compute_relative_bias(N, R, weights):
    """Compute the relative bias

    :param N: Non-representative data set
    :param R: Representative data set
    :param weights: Sample weights
    :return: Relative biases
    """
    weighted_means = compute_weighted_means(N, weights)
    population_means = np.mean(R, axis=0)
    return (abs(weighted_means - population_means) / population_means) * 100


def calculate_rbf_gamma(aggregate_set):
    """Calculate the gamma for the RBF-kernel

    :param aggregate_set: Aggregated data set
    :return: Gamma
    """
    all_distances = pdist(aggregate_set, "euclid")
    sigma = np.median(all_distances)
    return 1 / (2 * (sigma**2))


def scale_df(df, columns):
    """Scale the data set

    :param df: Data set
    :param columns: Scaling columns
    :return: Scaled data set and scaler
    """
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
    """_summary_

    :param x: _description_
    :param y: _description_
    :param weights: _description_
    :param gamma: _description_, defaults to None
    :param x_x_rbf_matrix: _description_, defaults to None
    :param y_y_rbf_matrix: _description_, defaults to None
    :param x_y_rbf_matrix: _description_, defaults to None
    :return: _description_
    """
    if gamma is None:
        gamma = calculate_rbf_gamma(np.append(x, y, axis=0))
    return compute_weighted_maximum_mean_discrepancy(
        gamma, x, y, weights, x_x_rbf_matrix, y_y_rbf_matrix, x_y_rbf_matrix
    )


def compute_weighted_maximum_mean_discrepancy(
    gamma, n, r, weights, n_n_rbf_matrix=None, r_r_rbf_matrix=None, n_r_rbf_matrix=None
):
    """_summary_

    :param gamma: _description_
    :param n: _description_
    :param r: _description_
    :param weights: _description_
    :param n_n_rbf_matrix: _description_, defaults to None
    :param r_r_rbf_matrix: _description_, defaults to None
    :param n_r_rbf_matrix: _description_, defaults to None
    :return: _description_
    """
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
    """_summary_

    :param scaled_N: _description_
    :param scaled_R: _description_
    :param weights: _description_
    :param scaler: _description_
    :param scale_columns: _description_
    :param columns: _description_
    :param gamma: _description_
    :return: _description_
    """
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
    """_summary_

    :param N: _description_
    :param R: _description_
    :param columns: _description_
    :param weights: _description_
    :param label: _description_
    :return: _description_
    """
    y_true = R[label]
    clf = train_classifier_auroc(N[columns], N[label], weights)
    y_predictions = clf.predict_proba(R[columns])[:, 1]
    auroc_score = roc_auc_score(y_true, y_predictions)
    auprc = average_precision_score(y_true, y_predictions.round())

    return auroc_score, auprc


def compute_test_metrics_mrs(data, columns, calculate_roc=False, weights=None, cv=3):
    """_summary_

    :param data: _description_
    :param columns: _description_
    :param calculate_roc: _description_, defaults to False
    :param weights: _description_, defaults to None
    :param cv: _description_, defaults to 3
    :return: _description_
    """
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


def train_pu_classifier(X_train, y_train, class_weight="balanced"):
    """_summary_

    :param X_train: _description_
    :param y_train: _description_
    :param class_weight: _description_, defaults to "balanced"
    :return: _description_
    """
    clf = RandomForestClassifier(
        class_weight=class_weight, n_estimators=25, n_jobs=-1, max_depth=25
    )
    return clf.fit(X_train, y_train)


def interpolate_roc(y_test, y_predict):
    """_summary_

    :param y_test: _description_
    :param y_predict: _description_
    :return: _description_
    """
    interpolation_points = 250
    interpolated_fpr = np.linspace(0, 1, interpolation_points)
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    interpolated_tpr = np.interp(interpolated_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0
    return interpolated_fpr, interpolated_tpr


def train_classifier_auroc(X_train, y_train, weights=None, speedup=True, cv=3):
    """_summary_

    :param X_train: _description_
    :param y_train: _description_
    :param weights: _description_, defaults to None
    :param speedup: _description_, defaults to True
    :param cv: _description_, defaults to 3
    :return: _description_
    """
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
    """_summary_

    :param rocs: _description_
    :return: _description_
    """
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
    """_summary_

    :param interpolated_fpr: _description_
    :param interpolated_tpr: _description_
    :return: _description_
    """
    mean_fpr = np.mean(interpolated_fpr, axis=0)
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    std_tpr = np.std(interpolated_tpr, axis=0)
    return mean_fpr, mean_tpr, std_tpr
