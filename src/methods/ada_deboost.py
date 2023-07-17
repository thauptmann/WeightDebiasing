import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.special import xlogy

from utils.metrics import compute_test_metrics_ada_deboost, compute_test_metrics_mrs


y_codes = np.array([-1.0, 1.0])


def ada_deboost_weighting(N, R, columns, *args, **kwargs):
    weights_N = np.ones(len(N)) / len(N)
    weights_R = np.ones(len(R)) / len(R)
    concat_data = pd.concat([N, R])
    max_patience = 10
    best_auroc_difference = np.inf
    current_patience = 0

    epsilon = np.finfo(weights_N.dtype).eps
    i = 0

    while True:
        i += 1
        auroc_test = compute_test_metrics_mrs(
            concat_data,
            columns,
            weights=np.concatenate([weights_N, weights_R]),
            cv=5,
        )

        auroc_difference = np.abs(0.5 - auroc_test)
        if auroc_difference < best_auroc_difference:
            best_auroc_difference = auroc_difference
        else:
            current_patience += 1
        if current_patience == max_patience or best_auroc_difference == 0.0:
            break

        predictions = train_weighted_random_forest(
            concat_data[columns],
            concat_data["label"],
            weights=np.concatenate([weights_N, weights_R]),
        )
        predictions_N = predictions[: len(N), 1]
        weights_N = update_weights(weights_N, epsilon, predictions_N)

    return weights_N


def update_weights(weights_N, epsilon, predictions_N):
    predictions_N = np.clip(predictions_N, a_min=epsilon, a_max=None)
    y_coding = y_codes.take(predictions_N < 0.5)
    estimator_weight = -0.25 * xlogy(y_coding, predictions_N)
    weight_modificator = np.exp(estimator_weight)
    weights_N *= weight_modificator
    weights_N = weights_N / weights_N.sum()
    return weights_N


def train_weighted_random_forest(x, label, weights):
    random_forest = RandomForestClassifier(max_depth=3, n_jobs=-1)
    random_forest = random_forest.fit(x, label, sample_weight=weights)
    return random_forest.predict_proba(x)
