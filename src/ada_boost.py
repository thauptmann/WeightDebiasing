import math
from pathlib import Path
import random
from tqdm.auto import tqdm
import numpy as np

from utils.compute_propensity_scores import propensity_scores
from utils.training import weighted_auc_prediction, cv_bootstrap_prediction, logistic_regression_prediction, \
    neural_network_prediction
from sklearn.metrics import roc_auc_score
from utils.visualisation import plot_line, plot_feature_distribution, plot_feature_histograms, plot_weights, \
    plot_probabilities, plot_asams
from utils.metrics import average_standardised_absolute_mean_distance, weighted_maximum_mean_discrepancy, \
    maximum_mean_discrepancy, scale_df


np.random.seed(0)
random.seed(0)


def ada_boost_debiasing(df, columns, dataset, number_of_splits=5, number_of_iterations=20, flip_rate=0.1):
    aurocs = list()
    boosted_aurocs = list()
    error_rates = list()
    asams = list()
    mmds = list()
    minimum_mmd = np.inf
    best_weights = None
    minimum_asam = np.inf
    eps = 1e-20
    method = 'adaboost'
    result_path = Path('..results')
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

    start_mmd = maximum_mean_discrepancy(scaled_R[columns], scaled_N[columns])
    mmds.append(start_mmd)
    asam = average_standardised_absolute_mean_distance(df, columns,
                                                       weights=np.ones(non_representative_size))
    asams.append(np.mean(asam))

    ratio = representative_size / non_representative_size
    weights = propensity_scores(scaled_df, columns, dataset, cv_bootstrap_prediction,
                                method='logistic_regression_classifier', )
    weights_non_representative = (np.ones(non_representative_size) / non_representative_size)
    weights_representative = np.ones(representative_size) / representative_size
    scaled_df['weights'] = 0
    normalized_weights = weights
    plot_weights(normalized_weights, weights_path, -1, 50)
    scaled_df.loc[scaled_df['label'] == 1, 'weights'] = normalized_weights
    scaled_df.loc[scaled_df['label'] == 0, 'weights'] = weights_representative

    for i in tqdm(range(number_of_iterations)):
        prediction_n, clf = cv_bootstrap_prediction(scaled_df, number_of_splits, columns)
        predicted_classes = np.round(prediction_n)
        flip = (np.random.uniform(0, 1, predicted_classes.shape)) < flip_rate
        predicted_classes = np.where(flip, (predicted_classes+1) % 1, predicted_classes)
        error_rate = np.count_nonzero(predicted_classes == np.ones_like(prediction_n)) / len(prediction_n)
        error_rates.append(error_rate)
        # error_rate = 0.7
        alpha = 0.5 * math.log(((1 - error_rate) / error_rate))
        samples_alpha = np.ones(non_representative_size) * -alpha
        samples_alpha[predicted_classes == 1] = alpha

        weights = scaled_df.weights[0:non_representative_size]
        plot_weights(weights, weights_path, i, 50)

        tmp = np.power(math.e, samples_alpha)
        new_weights = weights * tmp
        new_weights = new_weights / sum(new_weights)

        indices = list(range(0, non_representative_size))
        scaled_df.iloc[indices, scaled_df.columns.get_loc("weights")] = new_weights

        plot_probabilities(prediction_n, probabilities_path, i, 50)
        alpha_list.append(alpha)
        classifier_list.append(clf)
        auc, median_roc = weighted_auc_prediction(scaled_df, columns, i)
        aurocs.append(auc)

        prediction = np.zeros(len(scaled_df))
        half_list = int(len(alpha_list)/2)
        for alpha, classifier in zip(alpha_list[half_list:], classifier_list[half_list:]):
            weighted_prediction = classifier.predict_proba(scaled_df[columns])[:, 1] * alpha
            prediction += weighted_prediction
        prediction = prediction / sum(alpha_list)
        boosted_auroc = roc_auc_score(scaled_df['label'], prediction)
        boosted_aurocs.append(boosted_auroc)
        weighted_mmd = weighted_maximum_mean_discrepancy(scaled_N[columns], scaled_R[columns], new_weights.values)
        mmds.append(weighted_mmd)
        weighted_asams = average_standardised_absolute_mean_distance(df, columns, weights=new_weights)
        asams.append(np.mean(weighted_asams))
        if weighted_mmd < minimum_mmd:
            minimum_mmd = weighted_mmd
            best_weights = new_weights
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

    print(f'ASAM: {asams[0]}, {minimum_asam}')
    print(f'MMDs: {mmds[0]}, {minimum_mmd}')

    return best_weights
