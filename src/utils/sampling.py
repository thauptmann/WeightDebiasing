import numpy as np
import pandas as pd


def sample(bias_type, df, columns, bias_variable, bias_fraction=0.8):
    train = df.sample(frac=0.5, replace=False).copy()
    positive_samples = train[train[bias_variable] == 1]
    negative_samples = train[train[bias_variable] == 0]
    positive_fraction = 1
    negative_fraction = 1
    R = df.drop(train.index).copy().reset_index(drop=True)

    if bias_type == "less_positive_class":
        positive_fraction = positive_fraction - bias_fraction
        N = sample_N(
            positive_samples, negative_samples, positive_fraction, negative_fraction
        )
    elif bias_type == "less_negative_class":
        negative_fraction = negative_fraction - bias_fraction
        N = sample_N(
            positive_samples, negative_samples, positive_fraction, negative_fraction
        )
    elif bias_type == "mean_difference":
        mean_sample = train[columns].mean().values
        differences = np.linalg.norm(train[columns].values - mean_sample, axis=1) ** 4
        weight = -(1 / 20)
        sample_weights = np.exp(weight * differences)
        N = train.sample(frac=0.8, weights=sample_weights).reset_index(drop=True)
    else:
        N = train.reset_index(drop=True)

    N["label"] = 1
    R["label"] = 0

    return N, R


def sample_N(positive_samples, negative_samples, positive_fraction, negative_fraction):
    N = (
        pd.concat(
            [
                positive_samples.sample(frac=positive_fraction),
                negative_samples.sample(frac=negative_fraction),
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    return N
