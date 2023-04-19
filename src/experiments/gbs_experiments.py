import numpy as np
from pathlib import Path
from methods.domain_adaptation import calculate_rbf_gamma
from utils.metrics import compute_metrics, scale_df
import json
from utils.statistics import logistic_regression
from utils.visualisation import plot_gbs_results

eps = 1e-20


def gbs_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    loss_function=None,
    bias_sample_size=1000,
    bias_variable=None,
):
    result_path = Path("../results")
    if method == "neural_network_mmd_loss":
        method = f"{method}_{loss_function}"
    visualisation_path = (
        result_path / method / "folktables" / bias_variable / str(bias_sample_size)
    )
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

    weighted_mmd, weighted_ssmd, sample_biases, wasserstein_distances = compute_metrics(
        scaled_N[columns], scaled_R[columns], weights, scaler, scale_columns, gamma
    )

    result_dict = {"SSMD": weighted_ssmd, "MMDs": weighted_mmd}
    for column, bias in zip(
        scaled_df.drop(["pi"], axis="columns").columns, sample_biases
    ):
        result_dict[f"{column}_relative_bias"] = bias

    lr_pvalue_gbs, lr_pvalue_weighted = logistic_regression(
        N[columns + ["Wahlteilnahme"]], weights
    )

    result_dict["logistig regression p values"] = {
        "GBS": lr_pvalue_gbs,
        "Weighted": lr_pvalue_weighted,
    }
    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    bins = 100
    plot_gbs_results(
        bins,
        scaled_N,
        scaled_R,
        visualisation_path,
        weights,
    )

    N = df[df["label"] == 1]
