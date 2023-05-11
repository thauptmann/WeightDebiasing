import argparse
from experiments import (
    breast_cancer_experiment,
    census_experiment,
    folktables_experiment,
    gbs_allensbach_experiment,
    gbs_gesis_experiment,
    mrs_census_experiment,
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
    "high_bias",
]
dataset_list = [
    "gbs_allensbach",
    "gbs_gesis",
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
    "none",
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
    if dataset_name == "census":
        return census_experiment
    elif dataset_name == "folktables":
        return folktables_experiment
    elif dataset_name == "mrs_census":
        return mrs_census_experiment
    elif dataset_name == "breast_cancer":
        return breast_cancer_experiment
    elif dataset_name == "gbs_gesis":
        return gbs_gesis_experiment
    else:
        return gbs_allensbach_experiment
