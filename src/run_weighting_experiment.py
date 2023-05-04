import random
import torch
import numpy as np
from functools import partial

from utils.data_loader import load_dataset
from utils.input_arguments import input_arguments

from experiments.artificial_data_experiment import artificial_data_experiment
from experiments.census_experiments import census_experiments
from experiments.gbs_allensbach_experiments import gbs_allensbach_experiments
from experiments.folktables_experiments import folktables_experiments
from experiments.mrs_census_experiments import mrs_census_experiment
from experiments.breast_cancer_experiments import breast_cancer_experiment

from methods.logistic_regression import logistic_regression_weighting
from methods.naive_weighting import naive_weighting
from methods.neural_network_mmd_loss import neural_network_mmd_loss_weighting
from methods.neural_network_classifier import neural_network_weighting
from methods.random_forest import random_forest_weighting
from methods.gradient_boosting import gradient_boosting_weighting
from methods.ada_deboost import ada_deboost_weighting
from methods.domain_adaptation import domain_adaptation_weighting
from methods.maximum_representative_subsample import repeated_MRS
from methods.kmm import kernel_mean_matching


seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def weighting_experiment():
    args = input_arguments()
    dataset_name = args.dataset
    method_name = args.method
    bias_sample_size = args.bias_sample_size
    data, columns = load_dataset(dataset_name, args.bias_variable)

    compute_weights_function = get_weighting_function(method_name)

    if dataset_name == "artificial":
        artificial_data_experiment(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_sample_size=bias_sample_size,
            number_of_repetitions=args.number_of_repetitions,
        )
    elif dataset_name == "census":
        census_experiments(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_variable=args.bias_variable,
            bias_type=args.bias_type,
            bias_sample_size=bias_sample_size,
            number_of_repetitions=args.number_of_repetitions,
        )
    elif dataset_name == "folktables":
        folktables_experiments(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_variable=args.bias_variable,
            bias_type=args.bias_type,
            bias_sample_size=bias_sample_size,
            number_of_repetitions=args.number_of_repetitions,
        )
    elif dataset_name == "mrs_census":
        mrs_census_experiment(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_type=args.bias_type,
            bias_variable=args.bias_variable,
            number_of_repetitions=args.number_of_repetitions,
        )
    elif dataset_name == "breast_cancer":
        breast_cancer_experiment(
            data,
            columns,
            compute_weights_function,
            method=method_name,
            bias_type=args.bias_type,
            bias_variable=args.bias_variable,
            number_of_repetitions=args.number_of_repetitions,
        )
    else:
        gbs_allensbach_experiments(
            data,
            columns,
            compute_weights_function,
            method=method_name,
        )


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
    elif method_name == "adaDeBoost":
        compute_weights_function = ada_deboost_weighting
    elif method_name == "domain_adaptation":
        compute_weights_function = domain_adaptation_weighting
    elif method_name == "neural_network_mmd_loss_with_batches":
        compute_weights_function = partial(
            neural_network_mmd_loss_weighting, use_batches=True
        )
    elif method_name == "mrs":
        compute_weights_function = repeated_MRS
    elif method_name == "kmm":
        compute_weights_function = kernel_mean_matching
    else:
        compute_weights_function = naive_weighting

    return compute_weights_function


if __name__ == "__main__":
    weighting_experiment()
