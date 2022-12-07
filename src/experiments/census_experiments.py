import torch
import numpy as np
from pathlib import Path

from utils.data_loader import sample
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
    number_of_repetitions=3,
    bias_variable=None,
    bias_type=None,
    sample_size=2000,
    bias_strength=0.02,
    bias_sample_size=100,
    drop_duplicates=False,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "census" / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    equal_probability = 1 / len(df)
    equal_logit = np.log(equal_probability / (1 - equal_probability))

    if bias_type == "none":
        df["pi"] = equal_logit
    elif bias_type == "undersampling":
        df["pi"] = equal_logit - (df[bias_variable] * (equal_logit * bias_strength))
    elif bias_type == "oversampling":
        df["pi"] = equal_logit + (df[bias_variable] * (equal_logit * bias_strength))
    elif bias_type == "age":
        df["pi"] = ((200 - df["Age"]) ** 5) / ((200 - 10) ** 5)
    else:
        return

    odds = np.exp(df["pi"])
    df["pi"] = odds / (1 + odds)
    scaled_df, scaler = scale_df(df, columns)

    weighted_mmds_list = []
    asams_list = []
    biases_list = []
    mean_list = []
    mmd_list = []
    sample_mean_list = []

    for i in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, bias_sample_size, sample_size)
        if drop_duplicates:
            scaled_N = scaled_N.drop_duplicates()
        sample_means = np.mean(
            scaled_R.drop(["pi", "label"], axis="columns").values, axis=0
        )

        positive = np.sum(scaled_R[bias_variable].values)
        sample_representative_mean = positive / len(scaled_R)
        sample_mean_list.append(sample_representative_mean)

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
            df.drop(["pi"], axis="columns").columns,
            mean_biases,
            sd_biases,
        ):
            result_file.write(f"{column}: {bias} +- {sd}\n")

    if method == "neural_network_mmd_loss":
        plot_results_with_variance(
            mean_list,
            mmd_list,
            np.mean(sample_mean_list),
            visualisation_path,
        )
