import math
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
from src.utils.training import weighted_auc_prediction, cv_bootstrap_prediction
from sklearn.metrics import roc_auc_score
from src.utils.visualisation import plot_line, plot_feature_distribution, plot_feature_histograms, plot_weights, \
    plot_probabilities, plot_asams
from utils.metrics import average_standardised_absolute_mean_distance, weighted_maximum_mean_discrepancy, \
    maximum_mean_discrepancy, scale_df


def ada_boost_debiasing(df, columns, dataset, number_of_splits=5, number_of_iterations=20):
    method = 'adaboost'
    result_path = Path('../results')
    visualisation_path = result_path / method / dataset / 'visualisation'
    probabilities_path = result_path / method / dataset / 'probabilities'
    weights_path = result_path / 'adaboost' / dataset / 'weights'
    for path in (visualisation_path, probabilities_path, weights_path):
        path.mkdir(exist_ok=True, parents=True)
    alpha_list = list()
    classifier_list = list()
    scaled_df = scale_df(columns, df)
    scaled_N = scaled_df[scaled_df['label'] == 1]
    scaled_R = scaled_df[scaled_df['label'] == 0]
    non_representative_size = len(scaled_df[scaled_df['label'] == 1])
    representative_size = len(scaled_df[scaled_df['label'] == 0])
    weights_non_representative = np.ones(non_representative_size) / non_representative_size
    weights_representative = np.ones(representative_size) / representative_size
    scaled_df['weights'] = 0
    scaled_df.loc[scaled_df['label'] == 1, 'weights'] = weights_non_representative
    scaled_df.loc[scaled_df['label'] == 0, 'weights'] = weights_representative
    aurocs = list()
    boosted_aurocs = list()
    error_rates = list()
    asams = list()
    mmds = list()
    minimum_mmd = np.inf
    best_weights = None
    minimum_asam = np.inf

    start_mmd = maximum_mean_discrepancy(scaled_N[columns], scaled_R[columns])
    mmds.append(start_mmd)
    asam = average_standardised_absolute_mean_distance(scaled_df, columns,
                                                       weights=np.ones(non_representative_size))
    asams.append(np.mean(asam))

    for i in tqdm(range(number_of_iterations)):
        prediction_n, clf = cv_bootstrap_prediction(scaled_df, number_of_splits, columns)
        error_rate = np.count_nonzero(np.round(prediction_n) == np.ones_like(prediction_n)) / len(prediction_n)
        error_rates.append(error_rate)
        alpha = 0.5 * math.log(((1 - error_rate) / error_rate))
        samples_alpha = np.ones(non_representative_size) * -alpha
        samples_alpha[np.round(prediction_n) == 1] = alpha

        weights = scaled_df.weights[0:non_representative_size]
        plot_weights(weights, weights_path, i, 50)

        tmp = np.power(math.e, samples_alpha)
        new_weights = weights * tmp
        new_weights = new_weights / sum(new_weights)

        indices = list(range(0, non_representative_size))
        scaled_df.loc[indices, 'weights'] = new_weights

        plot_probabilities(prediction_n, probabilities_path, i, 50)
        alpha_list.append(alpha)
        classifier_list.append(clf)
        auc, median_roc = weighted_auc_prediction(scaled_df, columns, i)
        aurocs.append(auc)

        prediction = np.zeros(len(scaled_df))
        for alpha, classifier in zip(alpha_list, classifier_list):
            weighted_prediction = classifier.predict_proba(scaled_df[columns])[:, 1] * alpha
            prediction += weighted_prediction
        prediction = prediction / sum(alpha_list)
        boosted_auroc = roc_auc_score(scaled_df['label'], prediction)
        boosted_aurocs.append(boosted_auroc)
        weighted_mmd = weighted_maximum_mean_discrepancy(scaled_N[columns], scaled_R[columns],
                                                         scaled_df['weights'][0:non_representative_size])
        mmds.append(weighted_mmd)
        weighted_asams = average_standardised_absolute_mean_distance(scaled_df, columns,
                                                                     weights=scaled_df['weights'][
                                                                             0:non_representative_size])
        asams.append(np.mean(weighted_asams))
        if weighted_mmd < minimum_mmd:
            minimum_mmd = weighted_mmd
            best_weights = weights
            minimum_asam = np.mean(weighted_asams)

    plot_line(aurocs, visualisation_path, 'AUROC', plot_random_line=True)
    plot_line(boosted_aurocs, visualisation_path, 'Boosted AUROC', plot_random_line=True)
    plot_line(error_rates, visualisation_path, 'Total Error')
    plot_feature_distribution(scaled_df, columns, visualisation_path, best_weights)
    plot_feature_histograms(scaled_df, columns, visualisation_path, 50, best_weights)

    weighted_asams = average_standardised_absolute_mean_distance(scaled_df, columns, best_weights)
    plot_asams(weighted_asams, asam, columns, visualisation_path)
    plot_line(asams, visualisation_path, title='ASAM')
    plot_line(mmds, visualisation_path, title='MMD')

    print(f'{asams[0]}, {minimum_asam}')
    print(f'MMDs: {start_mmd}, {minimum_mmd}')
