import json
import numpy as np
from pathlib import Path

from utils.data_loader import sample_folk
from utils.visualisation import plot_weights

from utils.metrics import (
    compute_metrics,
    scale_df,
)
from utils.metrics import calculate_rbf_gamma
from tqdm import trange
from utils.visualisation import plot_results_with_variance



def folktables_experiments(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_variable=None,
    bias_sample_size=1000,
    loss_function=None,
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    if method == "neural_network_mmd_loss":
        method = f"{method}_{loss_function}"
    visualisation_path = (
        result_path / method / "folktables" / bias_variable / str(bias_sample_size)
    )
    visualisation_path.mkdir(exist_ok=True, parents=True)
    scale_columns = ["AGEP", "WKHP", "PINCP"]
    scaled_df, scaler = scale_df(df, scale_columns)

    country = scaled_df[scaled_df["label"] == 0]
    state = scaled_df[scaled_df["label"] == 1]

    gamma = calculate_rbf_gamma(scaled_df[columns][:1000])

    weighted_mmds_list = []
    biases_list = []
    mean_list = []
    mmd_list = []
    sample_mean_list = []
    dataset_ssmd_list = []
    parameter_ssmd_list = []
    wasserstein_parameter_list = []
    wasserstein_list = []

    for i in trange(number_of_repetitions):
        scaled_N, scaled_R = sample_folk(country, state, bias_sample_size)

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
            wasserstein_list=wasserstein_list,
            loss_function=loss_function,
        )

        if "mmd" in method:
            biases_path = visualisation_path / "biases"
            biases_path.mkdir(exist_ok=True)
            plot_results_with_variance(
                [mean_list[-1]],
                [mmd_list[-1]],
                [wasserstein_list[-1]],
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
        ) = compute_metrics(scaled_N[columns], scaled_R[columns], weights, scaler, scale_columns, gamma)

        plot_weights(weights, visualisation_path / "weights", i)

        biases_list.append(sample_biases)
        dataset_ssmd_list.append(np.nanmean(weighted_ssmd))
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
    
    for index, column in enumerate(scaled_df.drop(["label"], axis="columns").columns):
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
            wasserstein_list,
            np.mean(sample_mean_list),
            visualisation_path,
            True,
        )
