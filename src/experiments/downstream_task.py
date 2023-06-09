import json
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from tqdm import trange
from utils.sampling import sample

from utils.statistics import write_result_dict
from utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_metrics,
    calculate_rbf_gamma,
)
from utils.visualization import plot_results_with_variance, plot_weights


def downstream_experiment(
    df,
    columns,
    propensity_method,
    target,
    number_of_splits=10,
    method="",
    number_of_repetitions=100,
    bias_type=None,
    data_set_name="",
    **args
):
    weighted_mmds_list = []
    biases_list = []
    wasserstein_parameter_list = []
    mmd_list = []
    remaining_samples_list = []
    mean_list = []
    mse_list = []

    auroc_list = []
    accuracy_rate_list = []
    precision_list = []
    recall_list = []
    auprc_list = []
    tn_list = []
    fn_list = []
    tp_list = []
    fp_list = []

    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../../results")
    visualisation_path = result_path / method / "downstream" / data_set_name / bias_type
    visualisation_path.mkdir(exist_ok=True, parents=True)
    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns])
    sample_df = df.copy()

    for i in trange(number_of_repetitions):
        if len(df) > 5000:
            sample_df = df.sample(5000).copy()
        N, R = sample(bias_type, sample_df, columns, target)
        gamma = calculate_rbf_gamma(np.append(N[columns], R[columns], axis=0))

        weights = propensity_method(
            N,
            R,
            columns,
            save_path=visualisation_path,
            number_of_splits=number_of_splits,
            bias_variable=target,
            mean_list=mean_list,
            mmd_list=mmd_list,
            drop=1,
            early_stopping=True,
        )

        (
            auroc,
            accuracy,
            precision,
            recall,
            auprc,
            tn,
            fp,
            tree_fn,
            tree_tp,
        ) = compute_classification_metrics(N, R, columns, weights, target)

        if data_set_name == "folktables_income":
            mse = compute_regression_metrics(N, R, columns, weights, "Income")
        else:
            mse = 0

        (
            weighted_mmd,
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
        biases_list.append(sample_biases)
        wasserstein_parameter_list.append(wasserstein_distances)
        remaining_samples_list.append(remaining_samples)
        mse_list.append(mse)

        auroc_list.append(auroc)
        accuracy_rate_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        auprc_list.append(auprc)
        tn_list.append(tn)
        fn_list.append(tree_fn)
        tp_list.append(tree_tp)
        fp_list.append(fp)

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        wasserstein_parameter_list,
        remaining_samples_list,
        mse_list,
        auroc_list,
        accuracy_rate_list,
        precision_list,
        auprc_list,
        recall_list,
        tn_list,
        fn_list,
        tp_list,
        fp_list,
        len(N),
    )

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    if "neural_network_mmd_loss" in method:
        plot_results_with_variance(
            mmd_list,
            visualisation_path,
            "all",
        )
