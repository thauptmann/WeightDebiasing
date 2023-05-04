import numpy as np
from pathlib import Path
from methods.domain_adaptation import calculate_rbf_gamma
from utils.metrics import compute_metrics, scale_df
import json
from utils.statistics import logistic_regression
from utils.visualisation import (
    plot_gbs_results,
    plot_results_with_variance,
    plot_weights,
)

eps = 1e-20


def gbs_allensbach_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
):
    result_path = Path("../results")
    visualisation_path = result_path / method / "gbs_allensbach"
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.sample(frac=1)
    mmd_list = []
    mean_list = []
    scale_columns = columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    weights = propensity_method(
        scaled_N,
        scaled_R,
        columns,
        number_of_splits=number_of_splits,
        save_path=visualisation_path,
        mean_list=mean_list,
        mmd_list=mmd_list,
    )

    (
        weighted_mmd,
        weighted_ssmd,
        sample_biases,
        wasserstein_distances,
        weighted_ssmd_dataset,
    ) = compute_metrics(
        scaled_N,
        scaled_R,
        weights,
        scaler,
        scale_columns,
        columns,
        gamma,
    )

    remaining_samples = np.count_nonzero(weights != 0)
    plot_weights(weights, visualisation_path, 0, "GBS")

    if "neural_network_mmd_loss" in method:
        biases_path = visualisation_path / "MMDs"
        biases_path.mkdir(exist_ok=True)
        plot_results_with_variance(
            [],
            [mmd_list[-1]],
            [],
            biases_path,
            "",
        )

    result_dict = {
        "SSMD": weighted_ssmd_dataset,
        "MMDs": weighted_mmd,
        "Remaining Samples": remaining_samples,
        "Number of Samples": len(scaled_N)
    }
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias": sample_biases[index],
            "ssmd": weighted_ssmd[index],
            "wasserstein": wasserstein_distances[index],
        }

    lr_pvalue_gbs, lr_pvalue_weighted = logistic_regression(
        scaled_N[columns + ["Wahlteilnahme"]], weights
    )

    result_dict["logistig regression p values"] = {
        "GBS": lr_pvalue_gbs,
        "Weighted": lr_pvalue_weighted,
    }
    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    bins = 50
    plot_gbs_results(
        bins,
        scaled_N[columns],
        scaled_R[columns],
        visualisation_path,
        weights,
    )
