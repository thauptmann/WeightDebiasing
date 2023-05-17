import json
import time
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from tqdm import trange

from utils.statistics import write_result_dict
from utils.data_loader import sample_mrs_census
from utils.visualization import plot_weights, plot_results_with_variance
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_metrics,
    calculate_rbf_gamma,
)


def folktables_experiment(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_variable=None,
    bias_sample_size=1000,
    bias_type=None,
):
    weighted_mmds_list = []
    biases_list = []
    parameter_ssmd_list = []
    wasserstein_parameter_list = []
    mmd_list = []
    remaining_samples_list = []
    auroc_list = []
    mean_list = []
    accuracy_rate_list = []
    precision_list = []
    recall_list = []
    mse_list = []
    runtime_list = []

    bias_variable = "Binary Income"
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "folktables" / bias_variable / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)
    scaling_columns = ["AGEP", "WKHP"]
    scaler = StandardScaler()
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    for i in trange(number_of_repetitions):
        N, R = sample_mrs_census(bias_type, df, columns, "Binary Income", True)
        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        start_time = time.process_time()

        weights = propensity_method(
            N,
            R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=bias_variable,
            mean_list=mean_list,
            mmd_list=mmd_list,
            drop=1,
        )
        end_time = time.process_time()
        runtime = end_time - start_time

        if "neural_network_mmd_loss" in method:
            biases_path = visualisation_path / "MMDs"
            biases_path.mkdir(exist_ok=True)
            plot_results_with_variance(
                [],
                [mmd_list[-1]],
                [],
                biases_path,
                i,
            )

        (
            weighted_mmd,
            weighted_ssmd,
            sample_biases,
            wasserstein_distances,
        ) = compute_metrics(
            N,
            R,
            weights,
            scaler,
            scaling_columns,
            columns,
            gamma,
        )

        plot_weights(weights, visualisation_path / "weights", i)
        remaining_samples = np.count_nonzero(weights != 0)

        auroc, accuracy, precision, recall = compute_classification_metrics(
            N, R, columns, weights, "Binary Income"
        )
        mse = compute_regression_metrics(N, R, columns, weights, "Income")

        weighted_mmds_list.append(weighted_mmd)
        parameter_ssmd_list.append(weighted_ssmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)
        auroc_list.append(auroc)
        accuracy_rate_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        mse_list.append(mse)
        runtime_list.append(runtime)

    if "neural_network_mmd_loss" in method:
        biases_path = visualisation_path
        biases_path.mkdir(exist_ok=True)
        plot_results_with_variance(
            [],
            mmd_list,
            [],
            biases_path,
            "all",
        )

    result_dict = write_result_dict(
        weighted_mmds_list,
        biases_list,
        parameter_ssmd_list,
        wasserstein_parameter_list,
        remaining_samples_list,
        auroc_list,
        accuracy_rate_list,
        precision_list,
        recall_list,
        runtime_list,
        mse_list,
        N,
    )

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))
