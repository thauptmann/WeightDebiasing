import argparse
from utils.compute_propensity_scores import propensity_scores
from utils.data_loader import load_dataset, dataset_list
from utils.training import logistic_regression_prediction
from utils.statistics import logistic_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mainz", choices=dataset_list)
    parser.add_argument("--iterations", default=20, type=int)

    args = parser.parse_args()
    dataset_name = args.dataset
    data, columns = load_dataset(dataset_name)
    weights = propensity_scores(
        data,
        columns,
        dataset_name,
        logistic_regression_prediction,
        method="logistic_regression_classifier",
    )
    if dataset_name == "allensbach":
        N = data[data["label"] == 1]
        logistic_regression(N[columns + ["Wahlteilnahme"]], weights)
