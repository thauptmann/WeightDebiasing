import argparse
from .data_loader import dataset_list

method_list = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "neural_network_classifier",
    "neural_network_mmd_loss",
    "neural_network_mmd_loss_with_batches"
    "naive",
    "adaDebias",
    "domain_adaptation",
]


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allensbach", choices=dataset_list)
    parser.add_argument("--method", default="logistic_regression", choices=method_list)
    parser.add_argument("--iterations", default=20, type=int)

    return parser.parse_args()
