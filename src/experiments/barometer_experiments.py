import torch
import numpy as np
from pathlib import Path

from utils.data_loader import sample_barometer
from utils.visualisation import plot_weights

from utils.metrics import (
    average_standardised_absolute_mean_distance,
    compute_relative_bias,
    maximum_mean_discrepancy_weighted,
    scale_df,
    compute_weighted_means,
)
from tqdm import trange
import random

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def barometer_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    use_age_bias=None,
    sample_size=2000,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "barometer" / f"{use_age_bias=}"
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    df["pi"] = ((200 - df["age"]) ** 5) / ((200 - 10) ** 5)
    df["pi"] = np.exp(df["pi"]) / (1 + np.exp(df["pi"])).values
    scaled_df, scaler = scale_df(df, columns)

    weighted_mmds_list = []
    asams_list = []
    biases_list = []

    for i in trange(number_of_repetitions):
        scaled_N, scaled_R = sample_barometer(scaled_df, sample_size, use_age_bias)
        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=None,
        )

        weighted_mmd = maximum_mean_discrepancy_weighted(
            scaled_N[columns].values, scaled_R[columns].values, weights
        )
        weighted_asams = average_standardised_absolute_mean_distance(
            scaled_N, scaled_R, columns, weights
        )
        weighted_mmds_list.append(weighted_mmd)
        asams_list.append(np.nanmean(weighted_asams))

        scaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
        scaled_R[columns] = scaler.inverse_transform(scaled_R[columns])

        weighted_means = compute_weighted_means(
            scaled_N.drop(["pi", "label"], axis="columns"), weights
        )

        sample_means = np.mean(
            scaled_R.drop(["pi", "label"], axis="columns").values, axis=0
        )
        sample_biases = compute_relative_bias(weighted_means, sample_means)

        plot_weights(weights, visualisation_path / "weights", i)
        biases_list.append(sample_biases)

    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(
            f"ASAMS: {np.nanmean(asams_list)} +- {np.nanstd(asams_list)}\n"
        )
        result_file.write(
            f"MMDs: {np.nanmean(weighted_mmds_list)} +- "
            f"{np.nanstd(weighted_mmds_list)}\n\n"
        )
        result_file.write("\nRelative Biases:\n")
        for column, bias, sd in zip(
            scaled_N.drop(["pi"], axis="columns").columns,
            mean_biases,
            sd_biases,
        ):
            result_file.write(f"{column}: {bias} +- {sd}\n")
