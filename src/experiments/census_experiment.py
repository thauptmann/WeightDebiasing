import json
import numpy as np
from pathlib import Path

from utils.data_loader import sample
from utils.visualization import plot_weights

from utils.metrics import (
    compute_metrics,
    scale_df,
)
from utils.metrics import calculate_rbf_gamma
from tqdm import trange
from utils.visualization import plot_results_with_variance


def census_experiment(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_variable=None,
    bias_type=None,
    bias_strength=0.05,
    bias_sample_size=1000,
    loss_function=None,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = (
        result_path
        / method
        / "census"
        / bias_variable
        / bias_type
        / str(bias_sample_size)
    )
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df = df.reset_index(drop=True)

    equal_probability = 1 / len(df)
    equal_logit = np.log(equal_probability / (1 - equal_probability))

    if bias_type == "none":
        df["pi"] = equal_logit
    elif bias_type == "undersampling":
        df["pi"] = equal_logit - (df[bias_variable] * equal_logit * bias_strength)
    elif bias_type == "oversampling":
        df["pi"] = equal_logit + (df[bias_variable] * equal_logit * bias_strength)
    elif bias_type == "age":
        df["pi"] = ((200 - df["Age"]) ** 5) / ((200 - 10) ** 5)
    else:
        df["pi"] = equal_logit

    if bias_type != "age":
        odds = np.exp(df["pi"])
        df["pi"] = odds / (1 + odds)
    scale_columns = df.drop(["pi"], axis="columns").columns
    scaled_df, scaler = scale_df(df, scale_columns)

    gamma = calculate_rbf_gamma(scaled_df[columns])

    weighted_mmds_list = []
    biases_list = []
    mean_list = []
    mmd_list = []
    sample_mean_list = []
    dataset_ssmd_list = []
    parameter_ssmd_list = []
    wasserstein_parameter_list = []

    for i in trange(number_of_repetitions):
        scaled_N, scaled_R = sample(scaled_df, bias_sample_size)

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
            loss_function=loss_function,
        )

        if method == "neural_network_mmd_loss":
            biases_path = visualisation_path / "biases"
            biases_path.mkdir(exist_ok=True)
            plot_results_with_variance(
                [mean_list[-1]],
                [mmd_list[-1]],
                np.nanmean(sample_mean_list),
                biases_path,
                i,
                True,
            )

        (
            weighted_mmd,
            weighted_ssmd,
            sample_biases,
            wasserstein_distances,
            weighted_ssmd_dataset,
        ) = compute_metrics(
            scaled_N, scaled_R, weights, scaler, scale_columns, columns, gamma
        )

        plot_weights(weights, visualisation_path / "weights", i, bias_variable)

        biases_list.append(sample_biases)
        dataset_ssmd_list.append(np.mean(weighted_ssmd))
        parameter_ssmd_list.append(weighted_ssmd)
        weighted_mmds_list.append(weighted_mmd)
        wasserstein_parameter_list.append(wasserstein_distances)

    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)

    mean_ssmds = np.nanmean(parameter_ssmd_list, axis=0)
    sd_ssmds = np.nanstd(parameter_ssmd_list, axis=0)

    mean_wasserstein = np.nanmean(wasserstein_parameter_list, axis=0)
    sd_wasserstein = np.nanstd(wasserstein_parameter_list, axis=0)

    result_dict = {
        "SSMD": {
            "mean": np.nanmean(dataset_ssmd_list),
            "sd": np.nanstd(dataset_ssmd_list),
        },
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
    }
    for index, column in enumerate(scaled_df.drop(["pi"], axis="columns").columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
            "ssmd mean": mean_ssmds[index],
            "ssmd sd": sd_ssmds[index],
            "wasserstein mean": mean_wasserstein[index],
            "wasserstein sd": sd_wasserstein[index],
        }

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    if method == "neural_network_mmd_loss":
        plot_results_with_variance(
            mean_list,
            mmd_list,
            np.mean(sample_mean_list),
            visualisation_path,
            plot_mean=True,
        )
