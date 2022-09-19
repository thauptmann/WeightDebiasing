import argparse

from ada_boost import ada_boost_debiasing
from utils.data_loader import load_dataset, dataset_list
from utils.statistics import logistic_regression
from utils.input_arguments import input_arguments


if __name__ == '__main__':
    args = input_arguments()
    dataset_name = args.dataset
    data, columns = load_dataset(args.dataset)
    weights = ada_boost_debiasing(data, columns, number_of_iterations=args.iterations,
                                  dataset=dataset_name, flip_rate=args.flip_rate)
    #if args.dataset == 'allensbach':
    #    N = data[data['label'] == 1]
     #   logistic_regression(N[columns + ['Wahlteilnahme']], weights)
