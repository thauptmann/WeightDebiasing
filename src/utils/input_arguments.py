import argparse

method_list = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "neural_network_classifier",
    "neural_network_mmd_loss",
    "neural_network_mmd_loss_with_batches",
    "none",
    "adaDebias",
    "domain_adaptation",
    "random",
    "mrs",
    "kmm",
    "kliep",
]

bias_choice = [
    "oversampling",
    "undersampling",
    "none",
    "age",
    "complex",
    "less_negative_class",
    "less_positive_class",
    "none",
    "mean_difference",
]
dataset_list = ["gbs", "artificial", "census", "folktables", "mrs_census"]
bias_variables = [
    "Above_Below 50K",
    "Age",
    "",
    "PINCP",
]
loss_choices = ["mmd_rbf", "wasserstein", "mmd_linear"]


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=dataset_list, required=True)
    parser.add_argument("--method", choices=method_list, required=True)
    parser.add_argument("--bias", choices=bias_choice, default="none")
    parser.add_argument("--use_age_bias", action="store_true")
    parser.add_argument("--bias_sample_size", type=int, required=True)
    parser.add_argument(
        "--bias_variable", type=str, choices=bias_variables, default="Above_Below 50K"
    )
    parser.add_argument("--loss_choice", type=str, choices=loss_choices, default="mmd")
    parser.add_argument("--number_of_repetitions", default=100, type=int)

    return parser.parse_args()
