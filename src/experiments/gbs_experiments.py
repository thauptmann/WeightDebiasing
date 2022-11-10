import torch
import numpy as np
from pathlib import Path
from utils.metrics import (
    average_standardised_absolute_mean_distance,
    compute_relative_bias,
    maximum_mean_discrepancy_weighted,
    maximum_mean_discrepancy,
    scale_df,
    compute_weighted_means,
)
import random
from utils.visualisation import plot_results

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
eps = 1e-20


def gbs_experiments(
    df,
    columns,
    dataset,
    propensity_method,
    number_of_splits=10,
    bins=100,
    method="",
    bias_variable=None,
    bias_sample_size=0,
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

    weights = propensity_method(
        scaled_N,
        scaled_R,
        columns,
        save_path=visualisation_path,
        number_of_splits=number_of_splits,
        bias_variable=bias_variable,
    )

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

    scaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
    scaled_R[columns] = scaler.inverse_transform(scaled_R[columns])
    scaled_df[columns] = scaler.inverse_transform(scaled_df[columns])

    asams = [np.mean(asams_values), np.mean(weighted_asams)]

    weighted_means = compute_weighted_means(scaled_N, weights)
    population_means = np.mean(scaled_R.values, axis=0)
    relative_biases = compute_relative_bias(weighted_means, population_means)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(f"{asams=}\n")
        result_file.write(f"MMDs: {mmd}, {weighted_mmd}\n")
        result_file.write("\nRelative Bias:\n")
        for column, relative_bias in zip(scaled_df.columns, relative_biases):
            result_file.write(f"{column}: {relative_bias}\n")

    plot_results(
        bins,
        scaled_N,
        scaled_R,
        visualisation_path,
        weights,
    )

    return weights
