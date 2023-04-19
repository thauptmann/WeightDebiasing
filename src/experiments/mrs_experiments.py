import numpy as np
from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler
from methods.domain_adaptation import calculate_rbf_gamma
import pandas as pd
from utils.metrics import auc_prediction, compute_metrics
from tqdm import trange

from utils.visualisation import plot_results_with_variance, plot_weights

scaling_columns = [
    "Age",
    "Capital Loss",
    "Capital Gain",
    "Education Num",
    "Hours/Week",
]


census_bias = "Marital Status_Married"


def mrs_data_experiment(
    df,
    census_colums,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    loss_function="",
    bias_type="",
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "mrs_cenus" / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)

    train_df = df.sample(frac=0.9)
    test_df = df.drop(train_df.index)
    scaler = StandardScaler()
    train_df[scaling_columns] = scaler.fit_transform(train_df[scaling_columns])
    test_df[scaling_columns] = scaler.transform(test_df[scaling_columns])

    gamma = calculate_rbf_gamma(df[census_colums])

    weighted_mmds_list = []
    dataset_ssmd_list = []
    biases_list = []
    parameter_ssmd_list = []
    wasserstein_parameter_list = []
    mmd_list = []
    remaining_samples_list = []
    auroc_list = []
    bias_list = []

    for i in trange(number_of_repetitions):
        scaled_R, scaled_N = sample_data(bias_type, df, census_colums)

        weights = propensity_method(
            scaled_N,
            scaled_R,
            census_colums,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
            bias_variable=None,
            loss_function=loss_function,
            mmd_list=mmd_list,
        )

        indices_zero = (scaled_N[scaled_N["Marital Status_Married"] == 0]).index.values
        indices_one = (scaled_N[scaled_N["Marital Status_Married"] == 1]).index.values
        ratio = np.sum(weights[indices_one]) / np.sum(weights[indices_zero])
        ratio_rep = len(scaled_R[scaled_R["Marital Status_Married"] == 1]) / len(
            scaled_R[scaled_R["Marital Status_Married"] == 0]
        )
        bias = (abs(ratio - ratio_rep) / ratio_rep) * 100
        # auroc = auc_prediction(scaled_N, scaled_R, census_colums, weights)
        auroc = 0.5
        remaining_samples = np.count_nonzero(weights != 0)

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
            scaling_columns,
            census_colums,
            gamma,
        )

        dataset_ssmd_list.append(weighted_ssmd_dataset)
        weighted_mmds_list.append(weighted_mmd)
        parameter_ssmd_list.append(weighted_ssmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)
        auroc_list.append(auroc)
        bias_list.append(bias)

        plot_weights(weights, visualisation_path / "weights", i, bias_type)

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

    result_dict = write_result_dict(
        weighted_mmds_list,
        dataset_ssmd_list,
        biases_list,
        parameter_ssmd_list,
        wasserstein_parameter_list,
        remaining_samples_list,
        auroc_list,
        bias_list,
        scaled_N,
    )

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def write_result_dict(
    weighted_mmds_list,
    dataset_ssmd_list,
    biases_list,
    parameter_ssmd_list,
    wasserstein_parameter_list,
    remaining_samples_list,
    auroc_list,
    bias_list,
    scaled_N,
):
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
        "auroc": {
            "mean": np.nanmean(auroc_list),
            "sd": np.nanstd(auroc_list),
        },
        "bias": {
            "mean": np.nanmean(bias_list),
            "sd": np.nanstd(bias_list),
        },
        "remaining samples": {
            "mean": np.nanmean(remaining_samples_list),
            "sd": np.nanstd(remaining_samples_list),
        },
        "all_samples": len(scaled_N),
    }

    for index, column in enumerate(scaled_N.drop(["label"], axis="columns").columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
            "ssmd mean": mean_ssmds[index],
            "ssmd sd": sd_ssmds[index],
            "wasserstein mean": mean_wasserstein[index],
            "wasserstein sd": sd_wasserstein[index],
        }

    return result_dict


def sample_data(bias_type, df, columns):
    negative_normal = len(df[(df[census_bias] == 0)])
    positive_normal = len(df[(df[census_bias] == 1)])

    df_positive_class = df[(df[census_bias] == 1)]
    df_negative_class = df[(df[census_bias] == 0)]

    rep_fraction = 0.12
    bias_fraction = 0.05
    scaled_R = pd.concat(
        [
            df_negative_class.sample(int(negative_normal * 0.2)),
            df_positive_class.sample(int(positive_normal * 0.2)),
        ],
        ignore_index=True,
    )
    if bias_type == "less_positive_class":
        scaled_N = pd.concat(
            [
                df_negative_class.sample(int(negative_normal * rep_fraction)),
                df_positive_class.sample(
                    int(positive_normal * (rep_fraction - bias_fraction))
                ),
            ],
            ignore_index=True,
        )
    elif bias_type == "less_negative_class":
        scaled_N = pd.concat(
            [
                df_negative_class.sample(
                    int(negative_normal * (rep_fraction - bias_fraction))
                ),
                df_positive_class.sample(int(positive_normal * rep_fraction)),
            ],
            ignore_index=True,
        )
    elif bias_type == "mean_difference":
        mean_sample = df[columns].mean()
    else:
        scaled_N = pd.concat(
            [
                df_negative_class.sample(int(negative_normal * rep_fraction)),
                df_positive_class.sample(int(positive_normal * rep_fraction)),
            ],
            ignore_index=True,
        )

    scaled_R["label"] = 0
    scaled_N["label"] = 1
    return scaled_R, scaled_N
