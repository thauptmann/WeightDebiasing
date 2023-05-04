import numpy as np
from pathlib import Path
import json

from sklearn.preprocessing import StandardScaler
from methods.domain_adaptation import calculate_rbf_gamma
from utils.metrics import compute_classification_metrics, compute_metrics
from tqdm import trange
from utils.data_loader import sample_breast_cancer
from utils.statistics import write_result_dict

from utils.visualisation import plot_results_with_variance, plot_weights


weighted_mmds_list = []
dataset_ssmd_list = []
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


def breast_cancer_experiment(
    df,
    columns,
    propensity_method,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_type="",
    bias_variable="",
):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    if bias_type == "mean_difference":
        visualisation_path = result_path / method / "breast_cancer" / bias_type
    else:
        visualisation_path = result_path / method / "breast_cancer" / bias_variable
    visualisation_path.mkdir(exist_ok=True, parents=True)

    scaler = StandardScaler()
    scaler.fit(df[columns])

    for i in trange(number_of_repetitions):
        N, R = sample_breast_cancer(bias_variable, df, bias_type, columns)
        N[columns] = scaler.transform(N[columns])
        R[columns] = scaler.transform(R[columns])

        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        weights = propensity_method(
            N,
            R,
            columns,
            number_of_splits=number_of_splits,
            save_path=visualisation_path,
            bias_variable=bias_variable,
            mean_list=mean_list,
            mmd_list=mmd_list,
        )

        auroc, accuracy, precision, recall = compute_classification_metrics(
            N, R, columns, weights, "class"
        )
        remaining_samples = np.count_nonzero(weights != 0)

        (
            weighted_mmd,
            weighted_ssmd,
            sample_biases,
            wasserstein_distances,
            weighted_ssmd_dataset,
        ) = compute_metrics(
            N,
            R,
            weights,
            scaler,
            columns,
            columns,
            gamma,
        )

        dataset_ssmd_list.append(weighted_ssmd_dataset)
        weighted_mmds_list.append(weighted_mmd)
        parameter_ssmd_list.append(weighted_ssmd)
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)
        auroc_list.append(auroc)
        accuracy_rate_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        plot_weights(weights, visualisation_path / "weights", i, bias_type)

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
        dataset_ssmd_list,
        biases_list,
        parameter_ssmd_list,
        wasserstein_parameter_list,
        remaining_samples_list,
        auroc_list,
        accuracy_rate_list,
        precision_list,
        recall_list,
        N,
    )

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))
