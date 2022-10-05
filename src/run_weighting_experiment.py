from utils.compute_propensity_scores import propensity_scores
from utils.data_loader import load_dataset
from utils.input_arguments import input_arguments
from utils.propensity_scores_with_mmd_loss import neural_network_mmd_loss_prediction
from utils.statistics import logistic_regression
from utils.artificial_data_experiment import compute_artificial_data_metrics
from methods.logistic_regression import logistic_regression_weighting
from methods.naive_weighting import naive_weighting

args = input_arguments()
dataset_name = args.dataset
method_name = args.method
data, columns, bias_variable = load_dataset(dataset_name)

if method_name == "naive":
    compute_weights_function = naive_weighting
elif method_name == "logistic_regression":
    compute_weights_function = logistic_regression_weighting
elif method_name == "random_forest":
    pass
elif method_name == "gradient_boosting":
    pass
elif method_name == "neural_network_classifier":
    pass
elif method_name == "neural_network_mmd_loss":
    pass
elif method_name == "adaDebias":
    pass

if dataset_name == "artificial":
    compute_artificial_data_metrics(
        data,
        columns,
        dataset_name,
        compute_weights_function,
        method=method_name,
    )
else:
    weights = propensity_scores(
        data,
        columns,
        dataset_name,
        neural_network_mmd_loss_prediction,
        method="neural_network_mmd_loss",
        compute_weights=False,
        bias_variable=bias_variable,
    )

if dataset_name == "allensbach":
    N = data[data["label"] == 1]
    logistic_regression(N[columns + ["Wahlteilnahme"]], weights)
