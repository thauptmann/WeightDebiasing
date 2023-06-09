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

    tree_auroc_list = []
    tree_accuracy_rate_list = []
    tree_precision_list = []
    tree_recall_list = []
    tree_f_score_list = []
    tree_tn_list = []
    tree_fn_list = []
    tree_tp_list = []
    tree_fp_list = []

    svc_auroc_list = []
    svc_accuracy_rate_list = []
    svc_precision_list = []
    svc_recall_list = []
    svc_f_score_list = []
    svc_tn_list = []
    svc_fn_list = []
    svc_tp_list = []
    svc_fp_list = []

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
            tree_auroc,
            tree_accuracy,
            tree_precision,
            tree_recall,
            tree_f_score,
            tree_tn,
            tree_fp,
            tree_fn,
            tree_tp,
        ) = compute_classification_metrics(N, R, columns, weights, target)

        (
            svc_auroc,
            svc_accuracy,
            svc_precision,
            svc_recall,
            svc_f_score,
            svc_tn,
            svc_fp,
            svc_fn,
            svc_tp,
        ) = compute_classification_metrics(N, R, columns, weights, target, gamma)

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

        tree_auroc_list.append(tree_auroc)
        tree_accuracy_rate_list.append(tree_accuracy)
        tree_precision_list.append(tree_precision)
        tree_recall_list.append(tree_recall)
        tree_f_score_list.append(tree_f_score)
        tree_tn_list.append(tree_tn)
        tree_fn_list.append(tree_fn)
        tree_tp_list.append(tree_tp)
        tree_fp_list.append(tree_fp)

        svc_auroc_list.append(svc_auroc)
        svc_accuracy_rate_list.append(svc_accuracy)
        svc_precision_list.append(svc_precision)
        svc_recall_list.append(svc_recall)
        svc_f_score_list.append(svc_f_score)
        svc_tn_list.append(svc_tn)
        svc_fn_list.append(svc_fn)
        svc_tp_list.append(svc_tp)
        svc_fp_list.append(svc_fp)

    result_dict = write_result_dict(
        N.drop(["label"], axis="columns").columns,
        weighted_mmds_list,
        biases_list,
        wasserstein_parameter_list,
        remaining_samples_list,
        mse_list,
        tree_auroc_list,
        tree_accuracy_rate_list,
        tree_precision_list,
        tree_f_score_list,
        tree_recall_list,
        tree_tn_list,
        tree_fn_list,
        tree_tp_list,
        tree_fp_list,
        svc_auroc_list,
        svc_accuracy_rate_list,
        svc_precision_list,
        svc_recall_list,
        svc_f_score_list,
        svc_tn_list,
        svc_fn_list,
        svc_tp_list,
        svc_fp_list,
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
