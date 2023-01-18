from functools import partial

from utils.data_loader import load_dataset
from utils.input_arguments import input_arguments
from utils.statistics import logistic_regression

from experiments.artificial_data_experiment import artificial_data_experiment
from experiments.census_experiments import census_experiments
from experiments.gbs_experiments import gbs_experiments
from experiments.barometer_experiments import barometer_experiments

from methods.logistic_regression import logistic_regression_weighting
from methods.naive_weighting import naive_weighting
from methods.neural_network_mmd_loss import neural_network_mmd_loss_weighting

from methods.neural_network_classifier import neural_network_weighting
from methods.random_forest import random_forest_weighting
from methods.gradient_boosting import gradient_boosting_weighting
from methods.ada_debiasing import ada_debiasing_weighting
from methods.domain_adaptation import domain_adaptation_weighting
from methods.random import random_weighting


def weighting_experiment():
    args = input_arguments()
    dataset_name = args.dataset
    method_name = args.method
    bias_sample_size = args.bias_sample_size
    data, columns, bias_variable = load_dataset(dataset_name)

    compute_weights_function = get_weighting_function(method_name)

    if dataset_name == "artificial":
        artificial_data_experiment(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_sample_size=bias_sample_size,
        )
    elif dataset_name == "census":
        bias = args.bias
        weights = census_experiments(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_variable=bias_variable,
            bias_type=bias,
            bias_sample_size=bias_sample_size,
        )
    elif dataset_name == "barometer":
        use_age_bias = args.use_age_bias
        weights = barometer_experiments(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            use_age_bias=use_age_bias,
            bias_sample_size=bias_sample_size,
        )
    else:
        weights = gbs_experiments(
            data,
            columns,
            dataset_name,
            compute_weights_function,
            method=method_name,
            bias_variable=bias_variable,
            bias_sample_size=bias_sample_size,
        )
        if dataset_name == "allensbach":
            N = data[data["label"] == 1]
            logistic_regression(N[columns + ["Wahlteilnahme"]], weights)


def get_weighting_function(method_name):
    if method_name == "none":
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
    elif method_name == "neural_network_mmd_loss_with_batches":
        compute_weights_function = partial(
            neural_network_mmd_loss_weighting, use_batches=True
        )
    elif method_name == "random":
        compute_weights_function = random_weighting
    else:
        compute_weights_function = naive_weighting

    return compute_weights_function


if __name__ == "__main__":
    weighting_experiment()
