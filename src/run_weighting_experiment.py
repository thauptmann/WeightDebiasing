from utils.compute_propensity_scores import propensity_scores
from utils.data_loader import load_dataset
from utils.input_arguments import input_arguments
from utils.artificial_data_experiment import artificial_data_experiment
from utils.statistics import logistic_regression
from utils.census_experiments import census_experiments

from methods.logistic_regression import logistic_regression_weighting
from methods.naive_weighting import naive_weighting
from methods.neural_network_mmd_loss import (
    neural_network_mmd_loss_weighting,
    neural_network_mmd_loss_weighting_with_batches,
)
from methods.neural_network_classifier import neural_network_weighting
from methods.random_forest import random_forest_weighting
from methods.gradient_boosting import gradient_boosting_weighting
from methods.ada_debiasing import ada_debiasing_weighting
from methods.domain_adaptation import domain_adaptation_weighting


def weighting_experiment():
    args = input_arguments()
    dataset_name = args.dataset
    method_name = args.method
    data, columns, bias_variable = load_dataset(dataset_name)

    compute_weights_function = get_weighting_function(method_name)

    if dataset_name == "artificial":
        artificial_data_experiment(
            data,
            columns,
            dataset_name,
            compute_weights_function,
            method=method_name,
        )
    elif dataset_name == "census":
        weights = census_experiments(
            data,
            columns,
            dataset_name,
            compute_weights_function,
            method=method_name,
            bias_variable=bias_variable,
        )
    else:
        weights = propensity_scores(
            data,
            columns,
            dataset_name,
            compute_weights_function,
            method=method_name,
            bias_variable=bias_variable,
        )
        if dataset_name == "allensbach":
            N = data[data["label"] == 1]
            logistic_regression(N[columns + ["Wahlteilnahme"]], weights)


def get_weighting_function(method_name):
    if method_name == "naive":
        compute_weights_function = naive_weighting
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
    elif method_name == "adaDebias":
        compute_weights_function = ada_debiasing_weighting
    elif method_name == "domain_adaptation":
        compute_weights_function = domain_adaptation_weighting
    elif method_name == 'neural_network_mmd_loss_with_batches':
        compute_weights_function = neural_network_mmd_loss_weighting_with_batches

    return compute_weights_function


if __name__ == "__main__":
    weighting_experiment()