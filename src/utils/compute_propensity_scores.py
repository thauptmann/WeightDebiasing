import torch
import numpy as np
from pathlib import Path
from .metrics import (
    average_standardised_absolute_mean_distance,
    compute_relative_bias,
    maximum_mean_discrepancy_weighted,
    maximum_mean_discrepancy,
    scale_df,
    compute_weighted_means,
)
import random
from .visualisation import plot_results

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
eps = 1e-20


def propensity_scores(
    df,
    columns,
    dataset,
    propensity_method,
    number_of_splits=10,
    bins=25,
    method="",
    compute_weights=True,
    bias_variable=None,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.sample(frac=1)
    scaled_df, scaler = scale_df(df, columns)
    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]
    non_representative_size = len(scaled_df[scaled_df["label"] == 1])
    representative_size = len(scaled_df[scaled_df["label"] == 0])
    scaled_df.loc[scaled_df["label"] == 1, "weights"] = (
        np.ones(non_representative_size) / non_representative_size
    )
    scaled_df.loc[scaled_df["label"] == 0, "weights"] = (
        np.ones(representative_size) / representative_size
    )

    probabilities = propensity_method(
        scaled_N,
        scaled_R,
        columns,
        save_path=visualisation_path,
        number_of_splits=number_of_splits,
        bias_variable=bias_variable,
    )
    indices = list(range(0, non_representative_size))
    weights = (
        ((1 - probabilities) / probabilities) if compute_weights else probabilities
    )

    scaled_df.iloc[indices, scaled_df.columns.get_loc("weights")] = weights

    mmd = maximum_mean_discrepancy(scaled_N[columns].values, scaled_R[columns].values)
    asams_values = average_standardised_absolute_mean_distance(
        scaled_N, scaled_R, columns
    )
    weighted_mmd = maximum_mean_discrepancy_weighted(
        scaled_N[columns].values, scaled_R[columns].values, weights
    )
    weighted_asams = average_standardised_absolute_mean_distance(
        scaled_N, scaled_R, columns, weights
    )

    asams = [np.mean(asams_values), np.mean(weighted_asams)]
    number_of_zero_weights = np.count_nonzero(weights == 0)
    scaled_df[columns] = scaler.inverse_transform(scaled_df[columns])
    weighted_means = compute_weighted_means(scaled_N, weights)
    population_means = np.mean(scaled_R.values, axis=0)
    mean_weighted_means = np.mean(weighted_means, axis=0)
    relative_biases = compute_relative_bias(weighted_means, population_means)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(f"{asams=}\n")
        result_file.write(f"MMDs: {mmd}, {weighted_mmd}\n")
        result_file.write(f"{number_of_zero_weights=}\n")
        result_file.write("\nRelative Bias:\n")
        for column, relative_bias in zip(scaled_df.columns, relative_biases):
            result_file.write(f"{column}: {relative_bias}\n")

    plot_results(
        asams,
        asams_values,
        bins,
        columns,
        mmd,
        scaled_N,
        scaled_R,
        visualisation_path,
        weighted_asams,
        weighted_mmd,
        weights,
    )

    return weights
