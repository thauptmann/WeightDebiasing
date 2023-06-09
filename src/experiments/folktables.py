import json
import time
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from tqdm import trange

from utils.statistics import write_result_dict
from utils.sampling import sample
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
    bias_type=None,
    **args
):
    # Initialize metric lists
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
    tn_list = []
    fn_list = []
    tp_list = []
    fp_list = []

    bias_variable = "Binary Income"
    visualisation_path = create_result_directory(method, bias_variable, bias_type)
    df = df.reset_index(drop=True)
    for i in trange(number_of_repetitions):
        sample_df = df.sample(n=5000)
        scaler = scale_data(sample_df, columns)
        N, R = sample(bias_type, sample_df, columns, bias_variable)
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
            drop=5,
            early_stopping=True,
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
            auroc,
            accuracy,
            precision,
            recall,
            tn,
            fp,
            fn,
            tp,
        ) = compute_classification_metrics(N, R, columns, weights, bias_variable)
        mse = compute_regression_metrics(N, R, columns, weights, "Income")

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
            columns,
            columns,
            gamma,
        )

        plot_weights(weights, visualisation_path / "weights", i)
        remaining_samples = np.count_nonzero(weights != 0)

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
        tn_list.append(tn)
        fn_list.append(fn)
        tp_list.append(tp)
        fp_list.append(fp)

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
        tn_list,
        fn_list,
        tp_list,
        fp_list,
        N,
    )

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


def create_result_directory(method, bias_variable, bias_type):
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "folktables" / bias_variable / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)
    return visualisation_path


def scale_data(df, scaling_columns):
    scaler = StandardScaler()
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])
    return scaler
