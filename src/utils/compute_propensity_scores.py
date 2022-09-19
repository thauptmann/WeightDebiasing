import torch
import numpy as np
from pathlib import Path
from .metrics import (
    average_standardised_absolute_mean_distance,
    maximum_mean_discrepancy_weighted,
    maximum_mean_discrepancy,
    scale_df,
)
import random
from .visualisation import (
    plot_feature_distribution,
    plot_feature_histograms,
    plot_asams,
    plot_probabilities,
    plot_weights,
    plot_line,
)

seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
eps = 1e-10


def propensity_scores(
    df,
    columns,
    dataset,
    propensity_method,
    number_of_splits=5,
    bins=25,
    method="",
    compute_weights=True,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.sample(frac=1)
    scaled_df, scaler = scale_df(columns, df)
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

    probabilities, _ = propensity_method(
        scaled_df, number_of_splits, columns, False, data_set_name=dataset
    )
    indices = list(range(0, non_representative_size))
    weights = (
        ((1 - probabilities) / probabilities) if compute_weights else probabilities
    )

    scaled_df.iloc[indices, scaled_df.columns.get_loc("weights")] = weights

    mmd = maximum_mean_discrepancy(scaled_N[columns].values, scaled_R[columns].values)
    asams_values = average_standardised_absolute_mean_distance(scaled_df, columns)
    weighted_mmd = maximum_mean_discrepancy_weighted(
        scaled_N[columns].values, scaled_R[columns].values, weights
    )
    weighted_asams = average_standardised_absolute_mean_distance(
        scaled_df, columns, weights
    )
    asams = [np.mean(asams_values), np.mean(weighted_asams)]
    number_of_zero_weights = np.count_nonzero(weights == 0)

    scaled_df[columns] = scaler.inverse_transform(scaled_df[columns])

    plot_results(
        asams,
        asams_values,
        bins,
        columns,
        mmd,
        probabilities,
        scaled_df,
        visualisation_path,
        weighted_asams,
        weighted_mmd,
        weights,
    )

    with open(result_path / method / "results.txt", "w") as result_file:
        result_file.write(f"{asams=}\n")
        result_file.write(f"MMDs: {mmd}, {weighted_mmd}\n")
        result_file.write(f"{number_of_zero_weights=}\n")
    return weights


def plot_results(
    asams,
    asams_values,
    bins,
    columns,
    mmd,
    probabilities,
    scaled_df,
    visualisation_path,
    weighted_asams,
    weighted_mmd,
    weights,
):
    plot_asams(weighted_asams, asams_values, columns, visualisation_path)
    plot_feature_distribution(scaled_df, columns, visualisation_path, weights)
    plot_feature_histograms(scaled_df, columns, visualisation_path, bins, weights)
    plot_probabilities(probabilities, visualisation_path, 0, bins)
    plot_line(asams, visualisation_path, title="ASAM")
    plot_line([mmd, weighted_mmd], visualisation_path, title="MMD")
    plot_weights(weights / sum(weights), visualisation_path, 0, bins)
