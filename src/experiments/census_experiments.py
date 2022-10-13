import torch
import numpy as np
from pathlib import Path

from utils.data_loader import sample

from utils.metrics import (
    average_standardised_absolute_mean_distance,
    compute_relative_bias,
    maximum_mean_discrepancy_weighted,
    maximum_mean_discrepancy,
    scale_df,
    compute_weighted_means,
)
from tqdm import trange
import random
from utils.visualisation import plot_results

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
eps = 1e-20


def census_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    bins=100,
    method="",
    number_of_repetitions=500,
    bias_variable=None,
    bias=None,
    sample_size=1000,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / "census" / bias
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    equal_probability = 1 / len(df)
    if bias == 'none':
        df["pi"] = equal_probability
    elif bias == 'undersampling':
        df["pi"] = equal_probability - (df[bias] * (equal_probability * 0.5))
    else:
        df["pi"] = equal_probability + (df[bias] * (equal_probability * 0.5))

    scaled_df, scaler = scale_df(df, columns)

    population_means = np.mean(df.drop(["pi"], axis="columns").values, axis=0)

    weighted_mmds_list = []
    asams_list = []
    relative_biases_list = []

    for _ in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, sample_size)

        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=bias_variable,
        )

        mmd = maximum_mean_discrepancy(
            scaled_N[columns].values, scaled_R[columns].values
        )
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

        asams = [np.mean(asams_values), np.mean(weighted_asams)]
        number_of_zero_weights = np.count_nonzero(weights == 0)
        scaled_df[columns] = scaler.inverse_transform(scaled_df[columns])
        weighted_means = compute_weighted_means(scaled_N, weights)
        population_means = np.mean(scaled_R.values, axis=0)
        relative_biases = compute_relative_bias(weighted_means, population_means)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(f"{asams=}\n")
        result_file.write(f"MMDs: {mmd}, {weighted_mmd}\n")
        result_file.write(f"{number_of_zero_weights=}\n")
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
