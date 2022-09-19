import argparse
from .data_loader import dataset_list


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='allensbach',
                        choices=dataset_list)
    parser.add_argument('--iterations', default=20, type=int)
    parser.add_argument('--flip_rate', default=0.1, type=float)

    return parser.parse_args()
