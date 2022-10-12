import math

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

number_of_iterations = 50
param_grid = {"max_depth": [1, 2], "min_samples_split": [2, 5]}


def ada_debiasing_weighting(N, R, columns, number_of_splits, *args, **kwargs):
    weight_relation = len(N) / len(R)
    weights_N = np.ones(len(N)) / len(N)
    weights_R = (np.ones(len(R)) / len(R)) * weight_relation
    learning_rate = 0.5

    for _ in tqdm(range(number_of_iterations)):
        prediction_n = train_weighted_tree(
            N, R, number_of_splits, columns, np.concatenate([weights_N, weights_R])
        )
        predicted_classes = np.round(prediction_n)
        probability_difference = np.abs(prediction_n - 0.5)

        alpha = learning_rate * np.log(
            ((1 - probability_difference) / probability_difference)
        )
        samples_alpha = np.ones(len(N)) * alpha
        samples_alpha[predicted_classes == 1] = (
            samples_alpha[predicted_classes == 1] * -1
        )

        tmp = np.power(math.e, samples_alpha)
        new_weights_N = weights_N * tmp
        new_weights_N = new_weights_N / sum(new_weights_N)
        if (new_weights_N == weights_N).all():
            break
        else:
            weights_N = new_weights_N
        learning_rate *= 0.99

    return weights_N


def train_weighted_tree(N, R, number_of_splits, columns, weights):
    x_train = pd.concat([N, R])
    clf = train_tree(x_train[columns], x_train.label, weights, number_of_splits)
    predictions = clf.predict_proba(N[columns])[:, 1]
    return predictions


def train_tree(X_train, y_train, weights, number_of_splits):
    weights = np.nan_to_num(weights)
    decision_stump = GridSearchCV(
        DecisionTreeClassifier(), param_grid, n_jobs=-1, cv=number_of_splits
    )
    decision_stump = decision_stump.fit(X_train, y_train, sample_weight=weights)
    return decision_stump
