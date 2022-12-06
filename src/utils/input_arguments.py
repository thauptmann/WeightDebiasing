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
]

bias_choice = ["oversampling", "undersampling", "none", "age"]
dataset_list = ["gbs", "artificial", "census", "barometer"]


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=dataset_list, required=True)
    parser.add_argument("--method", choices=method_list, required=True)
    parser.add_argument("--bias",  choices=bias_choice, required=True)
    parser.add_argument("--use_age_bias", action="store_true")
    parser.add_argument("--drop_duplicates", action="store_true")
    parser.add_argument("--bias_sample_size", type=int, required=True)

    return parser.parse_args()
