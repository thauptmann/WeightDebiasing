import math
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
from src.utils.training import weighted_auc_prediction, cv_bootstrap_prediction
from sklearn.metrics import roc_auc_score
from src.utils.visualisation import plot_auc, plot_feature_distribution, plot_feature_histograms
from utils.metrics import average_standardised_absolute_mean_distance


def ada_boost_debiasing(df, columns, dataset, number_of_splits=5, number_of_iterations=20, mmd_iteration=1):
    result_path = Path('results')
    visualisation_path = result_path / 'visualisation' / dataset
    visualisation_path.mkdir(exist_ok=True, parents=True)
    weights = np.ones(len(df)) / len(df)
    df['weights'] = weights
    alpha_list = list()
    classifier_list = list()
    non_representative_size = len(df[df['label'] == 1])
    aurocs = list()
    boosted_aurocs = list()
    total_errors = list()
    rocs = list()
    mmds = list()
    best_value = 0.5
    best_iteration = 0
    best_weights = None
    eps = 1e-10

    for i in tqdm(range(number_of_iterations)):
        prediction_n, prediction_r, clf = cv_bootstrap_prediction(df, number_of_splits, columns)

        prediction = np.append(prediction_r, prediction_n)
        # total_error = np.count_nonzero(prediction_n < 0.5) / len(prediction_n)
        total_error = np.count_nonzero(np.round(prediction) != df.label) / len(prediction)
        # total_error = abs(0.5 - total_error)
        # prediction = np.append(prediction_n, prediction_r)
        # total_error = abs(0.5 - roc_auc_skcore(df.label, prediction))
        total_errors.append(total_error)
        alpha = 0.5 * math.log(((1 - total_error)/total_error)+eps)
        samples_alpha = np.ones(non_representative_size) * -alpha
        samples_alpha[np.round(prediction_n) != 0] = alpha
        # samples_alpha = (prediction_n - 0.5) * alpha
        if not i == number_of_iterations-1:
            tmp = np.power(math.e, samples_alpha)
            # weights[0:non_representative_size] = weights[0:non_representative_size] * tmp
            weights[0:non_representative_size] = weights[0:non_representative_size] * tmp
            weights = np.nan_to_num(weights)
            weights = weights / sum(weights)
            df['weights'] = weights

        alpha_list.append(alpha)
        classifier_list.append(clf)
        auc, median_roc = weighted_auc_prediction(df, columns, i)
        aurocs.append(auc)

        prediction = np.zeros(len(df))
        for alpha, classifier in zip(alpha_list, classifier_list):
            weighted_prediction = classifier.predict_proba(df[columns])[:, 1] * alpha
            prediction += weighted_prediction
        prediction = prediction / sum(alpha_list)
        boosted_auroc = roc_auc_score(df['label'], prediction)
        boosted_aurocs.append(boosted_auroc)
    plot_auc(aurocs, visualisation_path, 'AUROC', plot_random_line=False)
    plot_auc(boosted_aurocs, visualisation_path, 'Boosted AUROC', plot_random_line=False)
    plot_auc(total_errors, visualisation_path, 'Total Error')

    # df['weights'] = best_weights
    plot_feature_distribution(df, columns, visualisation_path)
    plot_feature_histograms(df, columns, visualisation_path)
    weighted = average_standardised_absolute_mean_distance(df, columns, weights=weights[0:non_representative_size])
    unweighted = average_standardised_absolute_mean_distance(df, columns, weights=np.ones(non_representative_size))
    print(f'weighted: {np.mean(weighted)}')
    print(f'unweighted: {np.mean(unweighted)}')
