import argparse

method_list = [
    "logistic_regression",
    "random_forest",
    "neural_network_mmd_loss",
    "none",
    "adaDeBoost",
    "random",
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
    "artificial",
    "census",
    "folktables",
    "mrs_census",
    "breast_cancer",
]

bias_variables = [
    "Above_Below 50K",
    "",
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
]

loss_choices = ["mmd_rbf", "wasserstein", "mmd_linear"]


def input_arguments():
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
