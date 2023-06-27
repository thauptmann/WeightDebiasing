import json
import numpy as np
from pathlib import Path
from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.statistics import logistic_regression
from utils.visualization import (
    plot_gbs_results,
    plot_results_with_variance,
    plot_weights,
)

bins = 50


def gbs_allensbach_experiment(
    df,
    columns,
    propensity_method,
    method,
    number_of_repetitions,
    number_of_splits=5,
    **args,
):
    result_path = Path("../results")
    visualisation_path = result_path / method / "gbs_allensbach"
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.sample(frac=1)
    mmd_list = []
    mean_list = []
    weighted_mmds_list = []
    biases_list = []
    wasserstein_parameter_list = []
    remaining_samples_list = []
    scale_columns = columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    for i in range(number_of_repetitions):
        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
            mean_list=mean_list,
            mmd_list=mmd_list,
            drop=1,
        )

        (
            weighted_mmd,
            sample_biases,
            wasserstein_distances,
        ) = compute_metrics(
            scaled_N, scaled_R, weights, scaler, scale_columns, columns, gamma
        )

        plot_weights(weights, visualisation_path / "weights", i)
        remaining_samples = np.count_nonzero(weights != 0)

        weighted_mmds_list.append(weighted_mmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)

        remaining_samples = np.count_nonzero(weights != 0)
        plot_weights(weights, visualisation_path, i, "GBS")

        if method == "neural_network_mmd_loss":
            biases_path = visualisation_path / "MMDs"
            biases_path.mkdir(exist_ok=True)
            plot_results_with_variance(
                [],
                [mmd_list[-1]],
                [],
                biases_path,
                "",
            )
        if method == "mrs":
            pass

    result_dict = {
        "MMDs": weighted_mmd,
        "Remaining Samples": remaining_samples,
        "Number of Samples": len(scaled_N),
    }
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias": sample_biases[index],
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

    plot_gbs_results(
        bins,
        scaled_N[columns],
        scaled_R[columns],
        visualisation_path,
        weights,
    )
