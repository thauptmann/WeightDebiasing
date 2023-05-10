import argparse
from experiments import (
    breast_cancer_experiment,
    census_experiments,
    folktables_experiments,
    gbs_allensbach_experiments,
    mrs_census_experiment,
)

from methods import (
    ada_deboost_weighting,
    domain_adaptation_weighting,
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
    "high_bias",
]
dataset_list = [
    "gbs_allensbach",
    "census",
    "folktables",
    "mrs_census",
    "breast_cancer",
]

bias_variables = [
    "Above_Below 50K",
    "Income",
    "Marital Status_Married",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
    "none"
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
    parser.add_argument("--bias_sample_size", type=int, default=1000)
    parser.add_argument(
        "--bias_variable", type=str, choices=bias_variables, default="Above_Below 50K"
    )
    parser.add_argument("--number_of_repetitions", default=100, type=int)

    return parser.parse_args()


def get_weighting_function(method_name):
    if method_name == "uniform":
        compute_weights_function = uniform_weighting
    elif method_name == "logistic_regression":
        compute_weights_function = logistic_regression_weighting
    elif method_name == "random_forest":
        compute_weights_function = random_forest_weighting
    elif method_name == "gradient_boosting":
        compute_weights_function = gradient_boosting_weighting
    elif method_name == "neural_network_classifier":
        compute_weights_function = neural_network_weighting
    elif method_name == "neural_network_mmd_loss":
        compute_weights_function = neural_network_mmd_loss_weighting
    elif method_name == "adaDeBoost":
        compute_weights_function = ada_deboost_weighting
    elif method_name == "domain_adaptation":
        compute_weights_function = domain_adaptation_weighting
    elif method_name == "mrs":
        compute_weights_function = repeated_MRS
    elif method_name == "kmm":
        compute_weights_function = kernel_mean_matching

    return compute_weights_function


def get_experiment_function(dataset_name):
    if dataset_name == "census":
        experiment_function = census_experiments
    elif dataset_name == "folktables":
        experiment_function = folktables_experiments
    elif dataset_name == "mrs_census":
        experiment_function = mrs_census_experiment
    elif dataset_name == "breast_cancer":
        experiment_function = breast_cancer_experiment
    else:
        experiment_function = gbs_allensbach_experiments
    return experiment_function
