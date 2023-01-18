import torch
import json
import numpy as np
from pathlib import Path
from methods.domain_adaptation import calculate_rbf_gamma

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
    sample_size=1000,
    bias_sample_size=1000,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = (
        result_path / method / "barometer" / f"{use_age_bias=}" / str(bias_sample_size)
    )
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    df["pi"] = ((200 - df["age"]) ** 5) / ((200 - 10) ** 5)
    df["pi"] = np.exp(df["pi"]) / (1 + np.exp(df["pi"])).values
    scale_columns = df.drop(["pi"], axis="columns").columns
    scaled_df, scaler = scale_df(df, scale_columns)
    gamma = calculate_rbf_gamma(scaled_df[columns])

    weighted_mmds_list = []
    asams_list = []
    biases_list = []

    for i in trange(number_of_repetitions):
        scaled_N, scaled_R = sample_barometer(
            scaled_df, sample_size, bias_sample_size, use_age_bias
        )
        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=None,
        )

        weighted_mmd = maximum_mean_discrepancy_weighted(
            scaled_N.drop(["pi", "label"], axis="columns").values,
            scaled_R.drop(["pi", "label"], axis="columns").values,
            weights,
            gamma,
        )
        weighted_asams = average_standardised_absolute_mean_distance(
            scaled_N.drop(["pi", "label"], axis="columns").values,
            scaled_R.drop(["pi", "label"], axis="columns").values,
            weights,
        )
        weighted_mmds_list.append(weighted_mmd)
        asams_list.append(np.nanmean(weighted_asams))

        scaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
        scaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])

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

    result_dict = {
        "ASAMS": {"mean": np.nanmean(asams_list), "sd": np.nanstd(asams_list)},
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
    }
    for column, bias, sd in zip(
        scaled_df.drop(["pi"], axis="columns").columns, mean_biases, sd_biases
    ):
        result_dict[f"{column}_relative_bias"] = {"mean": bias, "sd": sd}
    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))
