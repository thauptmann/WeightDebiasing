import torch
import numpy as np
from pathlib import Path
from .metrics import (
    average_standardised_absolute_mean_distance,
    maximum_mean_discrepancy_weighted,
    maximum_mean_discrepancy,
    scale_df,
    compute_relative_bias,
)
import random
from .visualisation import plot_results
from tqdm import trange

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
eps = 1e-20


def compute_artificial_data_metrics(
    df,
    columns,
    dataset,
    propensity_method,
    number_of_splits=10,
    bins=25,
    method="",
    number_of_repetitions=4,
    sample_size=1000,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    scaled_df, scaler = scale_df(df, columns)

    mmds_list = []
    asams_list = []
    relative_biases_list = []

    for _ in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, sample_size)
        scaled_N["weights"] = np.ones(sample_size) / sample_size
        scaled_R["weights"] = np.ones(sample_size) / sample_size

        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            number_of_splits=number_of_splits,
        )
        scaled_N["weights"] = weights

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

        asams = [np.mean(asams_values), np.mean(weighted_asams)]
        scaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
        scaled_R[columns] = scaler.inverse_transform(scaled_R[columns])
        relative_biases = compute_relative_bias(scaled_N, scaled_R, weights)

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

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(f"{asams=}\n")
        result_file.write(f"MMDs: {mmd}, {weighted_mmd}\n")
        result_file.write(f"{number_of_zero_weights=}\n")
        result_file.write("\nRelative Bias:\n")
        for column, relative_bias in zip(scaled_df.columns, relative_biases):
            result_file.write(f"{column}: {relative_bias}\n")
    return weights


def sample(df, sample_size):
    representative = df.sample(sample_size)
    representative["label"] = 0
    non_representative = df.sample(sample_size, weights=df["pi"])
    non_representative["label"] = 1
    return non_representative, representative
