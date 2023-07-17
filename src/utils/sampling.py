import numpy as np
import pandas as pd


def sample(
    bias_type, df, bias_variable, bias_fraction=0.1, train_fraction=0.25, columns=None
):
    train = df.sample(frac=train_fraction, replace=False).copy()
    positive_samples = train[train[bias_variable] == 1]
    negative_samples = train[train[bias_variable] == 0]
    R = df.drop(train.index).copy().reset_index(drop=True)

    if bias_type == "less_positive_class":
        N = sample_N(
            positive_samples,
            negative_samples,
            positive_fraction=bias_fraction,
            negative_fraction=1,
        )
    elif bias_type == "less_negative_class":
        N = sample_N(
            positive_samples,
            negative_samples,
            positive_fraction=1,
            negative_fraction=bias_fraction,
        )
    elif bias_type == "mean_difference":
        mean_sample = df[columns].mean().values
        differences = (
            np.linalg.norm(
                train[columns].values - mean_sample,
                axis=1,
            )
            ** 3
        )
        weight = -(1 / 20)
        sample_weights = np.exp(weight * differences)
        N = train.sample(frac=0.9, weights=sample_weights)
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
