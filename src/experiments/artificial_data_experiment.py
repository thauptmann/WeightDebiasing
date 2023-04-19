import numpy as np
from pathlib import Path
import json
from methods.domain_adaptation import calculate_rbf_gamma

from utils.data_loader import sample
from utils.metrics import compute_metrics, scale_df
from tqdm import trange

from utils.visualisation import plot_results_with_variance, plot_weights



def artificial_data_experiment(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_sample_size=1000,
    loss_function="",
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
    dataset_ssmd_list = []
    biases_list = []
    parameter_ssmd_list = []
    wasserstein_parameter_list = []

    for i in trange(number_of_repetitions):
        mmd_list = []
        scaled_N, scaled_R = sample(scaled_df, bias_sample_size)

        weights = propensity_method(
            scaled_N,
            scaled_R,
            columns,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
            bias_variable=None,
            loss_function=loss_function,
            mmd_list=mmd_list,
        )

        (
            weighted_mmd,
            weighted_ssmd,
            sample_biases,
            wasserstein_distances,
            weighted_ssmd_dataset,
        ) = compute_metrics(
            scaled_N.drop(["pi", "label"], axis="columns"),
            scaled_R.drop(["pi", "label"], axis="columns"),
            weights,
            scaler,
            scale_columns,
            columns,
            gamma,
        )

        dataset_ssmd_list.append(weighted_ssmd_dataset)
        weighted_mmds_list.append(weighted_mmd)
        parameter_ssmd_list.append(weighted_ssmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)

        plot_weights(weights, visualisation_path / "weights", i)

        if "neural_network_mmd_loss" in method:
            biases_path = visualisation_path / "biases"
            biases_path.mkdir(exist_ok=True)
            plot_results_with_variance(
                [],
                [mmd_list[-1]],
                [],
                biases_path,
                i,
            )

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

    for index, column in enumerate(
        scaled_N.drop(["label", "pi"], axis="columns").columns
    ):
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
            [],
            mmd_list,
            "",
            visualisation_path,
            False,
        )
