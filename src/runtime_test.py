from functools import partial
from pathlib import Path
from utils.data_loader import load_dataset
from utils.metrics import scale_df
from methods.logistic_regression import logistic_regression_weighting
from methods.naive_weighting import naive_weighting
from methods.neural_network_mmd_loss import neural_network_mmd_loss_weighting
from methods.neural_network_classifier import neural_network_weighting
from methods.random_forest import random_forest_weighting
from methods.gradient_boosting import gradient_boosting_weighting
from methods.ada_deboost import ada_deboost_weighting
from methods.domain_adaptation import domain_adaptation_weighting
from methods.random import random_weighting
from utils.input_arguments import method_list
from utils.data_loader import sample

import json
import timeit
from tqdm import tqdm


def runtime_experiment():
    dataset_name = "artificial"
    repeats = 100
    number_of_splits = 10
    bias_sample_size = 1000
    data, columns, _ = load_dataset(dataset_name)
    result_dict = {}

    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    save_path = result_path / "runtime"
    save_path.mkdir(exist_ok=True, parents=True)

    df = data.reset_index(drop=True)
    scale_columns = df.drop(["pi"], axis="columns").columns
    scaled_df, _ = scale_df(df, scale_columns)
    scaled_N, scaled_R = sample(scaled_df, bias_sample_size)

    for method_name in tqdm(method_list):
        method = get_weighting_function(method_name)

        time = timeit.timeit(
            lambda: method(
                scaled_N,
                scaled_R,
                columns,
                number_of_splits=number_of_splits,
                save_path="",
                bias_variable=None,
            ),
            number=repeats,
        )
        result_dict[method_name] = time / repeats

    with open(save_path / "runtimes.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))


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
        compute_weights_function = ada_deboost_weighting
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
    runtime_experiment()
