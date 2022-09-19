import argparse
from src.utils.compute_propensity_scores import propensity_scores
from src.utils.data_loader import load_dataset
from src.utils.input_arguments import input_arguments
from src.utils.statistics import logistic_regression
from src.utils.training import cv_bootstrap_prediction


if __name__ == '__main__':
    args = input_arguments()

    dataset_name = args.dataset
    data, columns = load_dataset(dataset_name)
    weights = propensity_scores(data, columns, dataset_name, cv_bootstrap_prediction,
                                method='random_forest_classifier')

    if dataset_name == 'allensbach':
        N = data[data['label'] == 1]
        logistic_regression(N[columns + ['Wahlteilnahme']], weights)
