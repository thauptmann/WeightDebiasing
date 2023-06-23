import argparse
from experiments import (
    gbs_allensbach_experiment,
    gbs_gesis_experiment,
    downstream_experiment,
)

from methods import (
    ada_deboost_weighting,
    gradient_boosting_weighting,
    kernel_mean_matching,
    logistic_regression_weighting,
    neural_network_mmd_loss_weighting,
    neural_network_weighting,
    random_forest_weighting,
    repeated_MRS,
    uniform_weighting,
)

method_list = [
    "logistic_regression",
    "random_forest",
    "neural_network_mmd_loss",
    "uniform",
    "adaDeBoost",
    "mrs",
    "kmm",
]

bias_choice = [
    "none",
    "less_negative_class",
    "less_positive_class",
    "mean_difference",
]
dataset_list = [
    "gbs_allensbach",
    "gbs_gesis",
    "census",
    "folktables",
    "folktables_income",
    "folktables_employment",
    "mrs_census",
    "breast_cancer",
    "hr_analytics",
    "loan_prediction",
]


mrs_ablation_experiments = [
    "random",
    "cross-validation",
    "temperature",
    "sampling",
    "class_weights",
]
down_stream_data_sets = [
    "breast_cancer",
    "folktables_employment",
    "folktables_income",
    "hr_analytics",
    "loan_prediction",
]


def parse_command_line_arguments():
    """Parses the command line arguments.

    Returns:
        _type_: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=dataset_list, required=True)
    parser.add_argument("--method", choices=method_list, required=True)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--number_of_repetitions", default=100, type=int)
    return parser.parse_args()


def parse_mrs_ablation_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_experiment", choices=mrs_ablation_experiments, required=True
    )
    parser.add_argument("--number_of_repetitions", default=100, type=int)
    parser.add_argument("--drop", default=1, type=int)
    return parser.parse_args()


def parse_mrs_analysis_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_name", choices=dataset_list, required=True)
    parser.add_argument("--number_of_repetitions", default=10, type=int)
    parser.add_argument("--bias_type", choices=bias_choice, default="none")
    parser.add_argument("--drop", default=1, type=int)

    return parser.parse_args()


def get_weighting_function(method_name):
    if method_name == "uniform":
        return uniform_weighting
    elif method_name == "logistic_regression":
        return logistic_regression_weighting
    elif method_name == "random_forest":
        return random_forest_weighting
    elif method_name == "gradient_boosting":
        return gradient_boosting_weighting
    elif method_name == "neural_network_classifier":
        return neural_network_weighting
    elif method_name == "neural_network_mmd_loss":
        return neural_network_mmd_loss_weighting
    elif method_name == "adaDeBoost":
        return ada_deboost_weighting
    elif method_name == "mrs":
        return repeated_MRS
    elif method_name == "kmm":
        return kernel_mean_matching


def get_experiment_function(dataset_name):
    if dataset_name == "gbs_gesis":
        return gbs_gesis_experiment
    elif dataset_name in down_stream_data_sets:
        return downstream_experiment
    else:
        return gbs_allensbach_experiment
