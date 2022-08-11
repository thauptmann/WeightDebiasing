import argparse

from src.adaBoost import ada_boost_debiasing
from src.utils.data_loader import load_dataset


def compute_weights(dataset_name, iterations):
    data, columns = load_dataset(dataset_name)
    ada_boost_debiasing(data, columns, number_of_iterations=iterations, dataset=dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mainz', choices=['allensbach', 'gesis', 'artificial'])
    parser.add_argument('--iterations', default=20, type=int)
    args = parser.parse_args()

    compute_weights(args.dataset, args.iterations)
