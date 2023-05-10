import random
import torch
import numpy as np

from utils.data_loader import load_dataset
from utils.command_line_arguments import (
    parse_command_line_arguments,
    get_weighting_function,
    get_experiment_function,
)

# Set random seeds for reproducibility.
seed = 5
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def weighting_experiment(
    dataset_name: str,
    method_name: str,
    bias_variable: str,
    bias_type: str,
    bias_sample_size: int,
    number_of_repetitions: int,
) -> None:
    """_summary_

    Args:
        dataset_name (str): Defines which data set to laod
        method_name (str): Defines which method is used
        bias_variable (str): Defines which variable will be biased
        bias_type (str): Defines the type of bias
        bias_sample_size (int): Size of bias sample
        number_of_repetitions (int): How many times the experiment is repeated
    """

    data, columns = load_dataset(dataset_name, bias_variable)
    compute_weights_function = get_weighting_function(method_name)
    experiment_function = get_experiment_function(dataset_name)

    experiment_function(
        data,
        columns,
        compute_weights_function,
        method=method_name,
        bias_variable=bias_variable,
        bias_type=bias_type,
        bias_sample_size=bias_sample_size,
        number_of_repetitions=number_of_repetitions,
        
    )


if __name__ == "__main__":
    args = parse_command_line_arguments()
    weighting_experiment(
        args.dataset,
        args.method,
        args.bias_variable,
        args.bias_type,
        args.bias_sample_size,
        args.number_of_repetitions,
    )
