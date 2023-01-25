import torch
import numpy as np
from pathlib import Path
from methods.domain_adaptation import calculate_rbf_gamma
from utils.metrics import (
    strictly_standardized_mean_difference,
    compute_relative_bias,
    maximum_mean_discrepancy_weighted,
    scale_df,
    compute_weighted_means,
)
import random
import json
from utils.visualisation import plot_gbs_results

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
    *args,
    **attributes,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.sample(frac=1)
    scale_columns = df.columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

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
        bias_variable=None,
    )

    weighted_mmd = maximum_mean_discrepancy_weighted(
        scaled_N.drop(["pi", "label"], axis="columns").values,
        scaled_R.drop(["pi", "label"], axis="columns").values,
        weights,
        gamma,
    )
    weighted_asams = strictly_standardized_mean_difference(
        scaled_N.drop(["pi", "label"], axis="columns").values,
        scaled_R.drop(["pi", "label"], axis="columns").values,
        weights,
    )

    scaled_N[scale_columns] = scaler.inverse_transform(scaled_N[scale_columns])
    scaled_R[scale_columns] = scaler.inverse_transform(scaled_R[scale_columns])
    scaled_df[scale_columns] = scaler.inverse_transform(scaled_df[scale_columns])

    weighted_means = compute_weighted_means(scaled_N, weights)
    population_means = np.mean(scaled_R.values, axis=0)
    relative_biases = compute_relative_bias(weighted_means, population_means)

    result_dict = {"ASAMS": weighted_asams, "MMDs": weighted_mmd}
    for column, bias in zip(
        scaled_df.drop(["pi"], axis="columns").columns, relative_biases
    ):
        result_dict[f"{column}_relative_bias"] = bias
    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    plot_gbs_results(
        bins,
        scaled_N,
        scaled_R,
        visualisation_path,
        weights,
    )

    return weights
