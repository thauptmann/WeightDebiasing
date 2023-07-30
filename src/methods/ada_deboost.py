import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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
    max_patience = 50
    best_auroc_difference = np.inf
    current_patience = 0

    param_grid = {"max_depth": [2, 3, 4]}
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid=param_grid,
        cv=2,
        n_jobs=-1,
        refit=False,
    )
    grid = grid.fit(
        concat_data[columns],
        concat_data["label"],
        sample_weight=np.concatenate([weights_N, weights_R]),
    )
    max_depth = grid.best_params_["max_depth"]

    while True:
        predictions = train_weighted_random_forest(
            concat_data[columns],
            concat_data["label"],
            weights=np.concatenate([weights_N, weights_R]),
            max_depth=max_depth,
        )
        auroc_test = roc_auc_score(concat_data["label"], predictions[:, 1])
        auroc_difference = np.abs(0.5 - auroc_test)
        if auroc_difference < best_auroc_difference:
            best_auroc_difference = auroc_difference
            best_weights = weights_N.copy()
        else:
            current_patience += 1
        if current_patience == max_patience or best_auroc_difference == 0.0:
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
    p_difference = np.abs(0.5 - predictions)
    y_coding = y_codes.take(predictions < 0.5)
    weight_modificator = y_coding * np.power(p_difference, 0.5) + 1
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
    random_forest = DecisionTreeClassifier(max_depth=max_depth, splitter="random")
    random_forest = random_forest.fit(x, label, sample_weight=weights)
    return random_forest.predict_proba(x)
