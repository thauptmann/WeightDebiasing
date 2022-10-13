import argparse

method_list = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "neural_network_classifier",
    "neural_network_mmd_loss",
    "neural_network_mmd_loss_with_batches",
    "naive",
    "adaDebias",
    "domain_adaptation",
]

bias_choice = ["oversampling", "undersampling", "none"]
dataset_list = ["gbs", "artificial", "census"]


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gbs", choices=dataset_list)
    parser.add_argument("--method", default="logistic_regression", choices=method_list)
    parser.add_argument("--bias", default="none", choices=bias_choice)

    return parser.parse_args()
