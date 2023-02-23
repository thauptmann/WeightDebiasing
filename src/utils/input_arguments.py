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
]

bias_choice = ["oversampling", "undersampling", "none", "age", "complex"]
dataset_list = ["gbs", "artificial", "census", "folktables"]
bias_variables = ["Above_Below 50K", "Age", ""]
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

    return parser.parse_args()
