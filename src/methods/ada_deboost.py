import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import calculate_rbf_gamma
from utils.weighted_mmd_loss import WeightedMMDLoss



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

    max_depth = 5
    gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))
    loss_function = WeightedMMDLoss(gamma, N[columns], R[columns])

    while True:
        predictions = train_weighted_random_forest(
            concat_data[columns],
            concat_data["label"],
            weights=np.concatenate([weights_N, weights_R]),
            max_depth=max_depth,
        )

        auroc_test = roc_auc_score(concat_data["label"], predictions[:, 1])
        auroc_difference = np.abs(0.5 - auroc_test)
        loss = loss_function(weights_N)
        print(loss.numpy(), flush=True)
        if loss < best_auroc_difference:
            best_auroc_difference = loss
            best_weights = weights_N.copy()
            current_patience = 0
        else:
            current_patience += 1
        if current_patience == max_patience:
            if max_depth > 1:
                max_depth -= 1
                current_patience = 0
                weights_N = best_weights.copy()
                continue
            else:
                break

        if best_auroc_difference == 0.0:
            break

        predictions_N = predictions[: len(N), 1]
        weights_N = update_weights(weights_N, predictions_N)

    return best_weights


def update_weights(weights, predictions):
    """Updates sample weights based on the prediction probabilities

    :param weights: The weights
    :param predictions: Prediction probabilities that are used to compute the
        new weights
    :return: Updated weights
    """
    epsilon = np.finfo(weights.dtype).eps
    predictions = np.clip(predictions, a_min=epsilon, a_max=None)
    weight_modificator = 1.5 - predictions
    weights *= weight_modificator
    weights = weights / weights.sum()
    return weights


def train_weighted_random_forest(x, label, weights, max_depth):
    """Trains a random forest and returns the predicted probabilties

    :param x: Training data
    :param label: Target label
    :param weights: Current weights
    :return: Predicted probabilities
    """
    random_forest = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=5,
        n_jobs=-1,
    )
    random_forest = random_forest.fit(x, label, sample_weight=weights)
    return random_forest.predict_proba(x)
