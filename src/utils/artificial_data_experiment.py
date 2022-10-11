import torch
import numpy as np
from pathlib import Path
from .metrics import (
    average_standardised_absolute_mean_distance,
    maximum_mean_discrepancy_weighted,
    scale_df,
    compute_weighted_means,
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


def artificial_data_experiment(
    df,
    columns,
    dataset,
    propensity_method,
    number_of_splits=10,
    bins=100,
    method="",
    number_of_repetitions=100,
    sample_size=1000,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)
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
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
        )

        weighted_mmd = maximum_mean_discrepancy_weighted(
            scaled_N[columns].values, scaled_R[columns].values, weights
        )
        weighted_asams = average_standardised_absolute_mean_distance(
            scaled_N, scaled_R, columns, weights
        )

        asams_list.append(weighted_asams)
        weighted_mmds_list.append(weighted_mmd)

        scaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
        scaled_R[columns] = scaler.inverse_transform(scaled_R[columns])
        weighted_means = compute_weighted_means(
            scaled_N.drop(["label", "pi"], axis="columns"), weights
        )

        relative_biases = compute_relative_bias(weighted_means, population_means)
        relative_biases_list.append(relative_biases)

    mean_biases = np.mean(relative_biases_list, axis=0)
    sd_biases = np.std(relative_biases_list, axis=0)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(f"ASAMS: {np.mean(asams_list)} +- {np.std(asams_list)}\n")
        result_file.write(
            f"MMDs: {np.mean(weighted_mmds_list)} +- {np.std(weighted_mmds_list)}\n"
        )
        result_file.write("\nBiases:\n")
        for column, mean_bias, sd_bias in zip(
            scaled_df.drop(["pi"], axis="columns").columns, mean_biases, sd_biases
        ):
            result_file.write(f"{column}: {mean_bias} +- {sd_bias}\n")

    plot_results(
        bins,
        scaled_N,
        scaled_R,
        visualisation_path,
        weights,
    )


def sample(df, sample_size):
    representative = df.sample(sample_size)
    representative["label"] = 0
    non_representative = df.sample(sample_size, weights=df["pi"])
    non_representative["label"] = 1
    return non_representative, representative
