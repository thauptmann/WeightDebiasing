import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from utils.metrics import compute_test_metrics_mrs


y_codes = np.array([-1.0, 1.0])


def ada_deboost_weighting(N, R, columns, *args, **kwargs):
    """Soft MRS method

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Sample weights
    """
    weights_N = np.ones(len(N)) / len(N)
    weights_R = np.ones(len(R)) / len(R)
    concat_data = pd.concat([N, R])
    max_patience = 10
    best_auroc_difference = np.inf
    current_patience = 0

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
        weights_N = update_weights(weights_N, predictions_N)

    return weights_N


def update_weights(weights, predictions):
    """Updates sample weights based on the prediction probabilities

    :param weights: The weights
    :param predictions: Prediction probabilities that are used to compute the
        new weights
    :return: Updated weights
    """
    epsilon = np.finfo(weights.dtype).eps
    predictions = np.clip(predictions, a_min=epsilon, a_max=None)
    p_difference = np.abs(0.5 - predictions)
    y_coding = y_codes.take(predictions < 0.5)
    alpha = 0.75 * np.log(p_difference)
    weight_modificator = y_coding * np.exp(alpha) + 1
    weights *= weight_modificator
    weights = weights / weights.sum()
    return weights


def train_weighted_random_forest(x, label, weights):
    """Trains a random forest and returns the predicted probabilties

    :param x: Training data
    :param label: Target label
    :param weights: Current weights
    :return: Predicted probabilities
    """
    random_forest = RandomForestClassifier(max_depth=2, n_jobs=-1)
    random_forest = random_forest.fit(x, label, sample_weight=weights)
    return random_forest.predict_proba(x)
