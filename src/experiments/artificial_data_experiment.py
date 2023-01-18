import torch
import numpy as np
from pathlib import Path
import json
from methods.domain_adaptation import calculate_rbf_gamma

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
    method="",
    number_of_repetitions=100,
    bias_sample_size=1000,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "artificial" / str(bias_sample_size)
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)
    scale_columns = df.drop(["pi"], axis="columns").columns
    scaled_df, scaler = scale_df(df, scale_columns)
    gamma = calculate_rbf_gamma(scaled_df[columns])

    weighted_mmds_list = []
    asams_list = []
    biases_list = []

    for _ in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, bias_sample_size)
        

        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
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

        asams_list.append(weighted_asams)
        weighted_mmds_list.append(weighted_mmd)

        scaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
        scaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])
        weighted_means = compute_weighted_means(
            scaled_N.drop(["label", "pi"], axis="columns"), weights
        )

        sample_means = np.mean(
            scaled_R.drop(["pi", "label"], axis="columns").values, axis=0
        )
        sample_biases = compute_relative_bias(weighted_means, sample_means)

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
