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
from utils.visualisation import plot_results_with_variance

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def census_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=1,
    bias_variable=None,
    bias_type=None,
    sample_size=2000,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / "census" / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    equal_probability = 1 / len(df)
    bias_strength = 0.4
    if bias_type == "none":
        df["pi"] = equal_probability
    elif bias_type == "undersampling":
        df["pi"] = equal_probability - (
            df[bias_variable] * (equal_probability * bias_strength)
        )
    else:
        df["pi"] = equal_probability + (
            df[bias_variable] * (equal_probability * bias_strength)
        )

    scaled_df, scaler = scale_df(df, columns)

    population_means = np.mean(df.drop(["pi"], axis="columns").values, axis=0)
    positive = np.sum(df[bias_variable].values)
    representative_mean = positive / len(df)

    weighted_mmds_list = []
    asams_list = []
    relative_biases_list = []
    mean_list = []
    mmd_list = []

    for _ in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, sample_size)

        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=bias_variable,
            mean_list=mean_list,
            mmd_list=mmd_list,
        )

        weighted_mmd = maximum_mean_discrepancy_weighted(
            scaled_N[columns].values, scaled_R[columns].values, weights
        )
        weighted_asams = average_standardised_absolute_mean_distance(
            scaled_N, scaled_R, columns, weights
        )
        weighted_mmds_list.append(weighted_mmd)
        asams_list.append(np.mean(weighted_asams))

        scaled_N[columns] = scaler.inverse_transform(scaled_N[columns])
        scaled_R[columns] = scaler.inverse_transform(scaled_R[columns])

        weighted_means = compute_weighted_means(
            scaled_N.drop(["pi", "label"], axis="columns"), weights
        )
        relative_biases = compute_relative_bias(weighted_means, population_means)
        relative_biases_list.append(relative_biases)

    mean_biases = np.nanmean(relative_biases_list, axis=0)
    sd_biases = np.nanstd(relative_biases_list, axis=0)

    with open(visualisation_path / "results.txt", "w") as result_file:
        result_file.write(
            f"ASAMS: {np.nanmean(asams_list)} +- {np.nanstd(asams_list)}\n"
        )
        result_file.write(
            f"MMDs: {np.nanmean(weighted_mmds_list)} +- "
            f"{np.nanstd(weighted_mmds_list)}\n"
        )
        result_file.write("\nRelative Biases:\n")
        for column, mean_bias, sd_bias in zip(
            scaled_df.drop(["pi"], axis="columns").columns, mean_biases, sd_biases
        ):
            result_file.write(f"{column}: {mean_bias} +- {sd_bias}\n")

    if method == "neural_network_with_mmd_loss":
        plot_results_with_variance(
            mean_list,
            mmd_list,
            representative_mean,
            visualisation_path,
        )
