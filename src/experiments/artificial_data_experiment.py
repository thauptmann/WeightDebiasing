import torch
import numpy as np
from pathlib import Path

from utils.data_loader import sample
from utils.metrics import (
    average_standardised_absolute_mean_distance,
    maximum_mean_discrepancy_weighted,
    scale_df,
    compute_weighted_means,
    compute_relative_bias,
)
import random
from tqdm import trange

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def artificial_data_experiment(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    bins=100,
    method="",
    number_of_repetitions=500,
    sample_size=1000,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "artificial"
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)
    scaled_df, scaler = scale_df(df, columns)
    bias_variable = None

    weighted_mmds_list = []
    asams_list = []
    weighted_means_list = []

    population_means = np.mean(df.drop(["pi"], axis="columns").values, axis=0)

    for _ in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, sample_size)
        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
            bias_variable=bias_variable,
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

        weighted_means_list.append(weighted_means)

    weighted_means = np.mean(weighted_means_list, axis=0)
    relative_biases = compute_relative_bias(weighted_means, population_means)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(
            f"ASAMS: {np.nanmean(asams_list)} +- {np.nanstd(asams_list)}\n"
        )
        result_file.write(
            f"MMDs: {np.nanmean(weighted_mmds_list)} +- "
            f"{np.nanstd(weighted_mmds_list)}\n"
        )
        result_file.write("\nRelative Biases:\n")
        for column, bias in zip(
            scaled_df.drop(["pi"], axis="columns").columns, relative_biases
        ):
            result_file.write(f"{column}: {bias} \n")
