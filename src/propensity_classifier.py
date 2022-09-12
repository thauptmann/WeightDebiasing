import argparse
import math
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
import numpy as np

from src.utils.data_loader import load_dataset
from src.utils.training import train_classifier
from src.utils.visualisation import plot_feature_distribution, plot_feature_histograms, plot_asams
from utils.metrics import average_standardised_absolute_mean_distance, weighted_maximum_mean_discrepancy, \
    maximum_mean_discrepancy

eps = 1e-10


def propensity_scores(df, columns, dataset):
    result_path = Path('../results')
    visualisation_path = result_path / 'classifier' / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    non_representative_size = len(df[df['label'] == 1])

    clf = RandomForestClassifier(n_estimators=50, max_depth=5)
    probabilities = train_classifier(df, clf, columns)

    df['weights'] = (1 - probabilities) / probabilities
    plot_feature_distribution(df, columns, visualisation_path)
    plot_feature_histograms(df, columns, visualisation_path)
    x = df[df['label'] == 1]
    y = df[df['label'] == 0]
    # weighted = weighted_maximum_mean_discrepancy(x[columns], y[columns], x['weights'])
    # unweighted = maximum_mean_discrepancy(x[columns], y[columns])
    weighted_asams  = average_standardised_absolute_mean_distance(df, columns,
                                                           weights=df['weights'][0:non_representative_size])
    unweighted_asams = average_standardised_absolute_mean_distance(df, columns, weights=np.ones(non_representative_size))
    # print(f'weighted: {np.mean(weighted)}')
    # print(f'unweighted: {np.mean(unweighted)}')
    plot_asams(weighted_asams, unweighted_asams, columns, visualisation_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mainz', choices=['allensbach', 'gesis', 'artificial'])
    parser.add_argument('--iterations', default=20, type=int)

    args = parser.parse_args()
    dataset_name = args.dataset
    data, columns = load_dataset(dataset_name)
    propensity_scores(data, columns, dataset_name)
