from src.utils.compute_propensity_scores import propensity_scores
from src.utils.data_loader import load_dataset
from src.utils.input_arguments import input_arguments
from src.utils.training import neural_network_prediction

if __name__ == '__main__':
    args = input_arguments()
    dataset_name = args.dataset
    data, columns = load_dataset(dataset_name)
    propensity_scores(data, columns, dataset_name, neural_network_prediction, method='neural_network_classifier')
