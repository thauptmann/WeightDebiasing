from utils.compute_propensity_scores import propensity_scores
from utils.data_loader import load_dataset
from utils.input_arguments import input_arguments
from utils.statistics import logistic_regression
from utils.training import neural_network_prediction

if __name__ == '__main__':
    args = input_arguments()
    dataset_name = args.dataset
    data, columns = load_dataset(dataset_name)
    weights = propensity_scores(data, columns, dataset_name, neural_network_prediction,
                                method='neural_network_classifier')

    if dataset_name == 'allensbach':
        N = data[data['label'] == 1]
        logistic_regression(N[columns + ['Wahlteilnahme']], weights)
