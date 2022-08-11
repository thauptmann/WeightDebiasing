import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_auc_median(auc_score, std_aucs, drop, file_name, number_of_samples, save, mrs_iterations):
    aucs_upper = np.minimum(auc_score + std_aucs, 1)
    aucs_younger = np.maximum(auc_score - std_aucs, 0)
    x_labels = range(number_of_samples, drop, -drop)
    plt.fill_between(x_labels, aucs_younger, aucs_upper, color='blue', alpha=0.2)
    plt.plot(x_labels, auc_score, color='blue', label='Median AUROC', linestyle='-')
    plt.plot((len(x_labels) - 1) * drop * [0.5], color='black', linestyle='--', label='Random')
    plt.ylabel('AUROC')
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    minimum = min(0.5, np.min(aucs_younger))
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    plt.vlines(mrs_iterations, minimum, maximum, colors='black', linestyles='solid')
    plt.xticks(list(range(number_of_samples, 0, -100)) + [0])
    plt.gca().invert_xaxis()
    plt.xlabel('Number of remaining samples')
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(list(mrs_iterations))
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_auc(auc_score, file_name,   title='', plot_random_line=False):
    plt.plot(auc_score, color='blue', linestyle='-', label=title)
    if plot_random_line:
        plt.plot(len(auc_score) * [0.5], color='black', linestyle='--', label='Random')
    plt.ylabel(title)
    plt.xlabel('Iteration')
    plt.savefig(file_name / f'{title}.pdf')
    plt.clf()


def plot_feature_distribution(df, columns, file_name):
    plot_directory = file_name / 'distributions'
    plot_directory.mkdir(exist_ok=True)
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    for column_name in columns:
        sns.ecdfplot(N, x=column_name,   label='Non-Representative')
        sns.ecdfplot(R, x=column_name,  label='Representative')
        sns.ecdfplot(N, x=column_name, weights=N.weights, label='Weighted')
        plt.legend(title='Data set')
        plt.savefig(plot_directory / f'{column_name}.pdf')
        plt.clf()


def plot_feature_histograms(df, columns, file_name):
    plot_directory = file_name / 'histograms'
    plot_directory.mkdir(exist_ok=True)
    N = df[df['label'] == 1]
    R = df[df['label'] == 0]
    for column_name in columns:
        fig, ax = plt.subplots(1, 3)
        unique = df[column_name].nunique()
        sns.histplot(N, x=column_name, ax=ax[0], bins=unique, stat="probability").set_title('Non-Representative')
        sns.histplot(R, x=column_name, ax=ax[1], bins=unique, stat="probability").set_title('Representative')
        sns.histplot(N, x=column_name, weights=N.weights, ax=ax[2], bins=unique, stat="probability").set_title('Weighted')
        fig.savefig(plot_directory / f'{column_name}.pdf')
        plt.clf()


def plot_rocs(start_roc, weighted_roc, file_name):
    for fper, tper, std, deleted_elements, name in zip((start_roc, weighted_roc), ('Start', 'Weighted')):
        tpfrs_higher = np.minimum(tper + std, 1)
        tpfrs_lower = np.maximum(tper - std, 0)
        plt.plot(fper, tper, label=f'{name}')
        plt.fill_between(fper, tpfrs_lower, tpfrs_higher, alpha=0.2)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def plot_mmds(mmds, drop, mmd_iteration, file_name, mrs_iterations, number_of_samples, save=False):
    x_labels = range(number_of_samples, 0, -drop * mmd_iteration)
    plt.plot(x_labels, mmds, linestyle='-')
    minimum = np.min(mmds)
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors='black', linestyles='solid')
    plt.ylabel('Maximum Mean Discrepancy')
    plt.xlabel('Number of remaining samples')
    plt.xticks(list(range(number_of_samples, 0, -500)) + [0])
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks([mrs_iterations])
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_mmds_median(mmds, std, drop, mmd_iteration, file_name, mrs_iterations,
                     number_of_samples, save=False):
    mmds_upper = np.minimum(mmds + std, 1)
    mmds_more_negativ = np.maximum(mmds - std, 0)
    x_labels = range(number_of_samples, drop * mmd_iteration, -drop * mmd_iteration)
    plt.fill_between(x_labels, mmds_more_negativ, mmds_upper, color='black', alpha=0.2)
    plt.plot(x_labels, mmds, linestyle='-')
    minimum = np.min(mmds_more_negativ)
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors='black', linestyles='solid')
    plt.ylabel('Maximum Mean Discrepancy')
    plt.xlabel('Number of remaining samples')
    plt.xticks(list(range(number_of_samples, 0, -100)) + [0])
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(mrs_iterations)
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_class_ratio(ratios, representative_ratio, file_name, mrs_iterations, number_of_samples, save=False):
    plt.xlabel('Number of remaining samples')
    plt.ylabel('Ratio of married persons')

    plt.plot(ratios, label='non-representative', linestyle='-', color='blue')
    plt.plot(number_of_samples * [representative_ratio], color='black', linestyle='--', label='representative')
    minimum = np.min(ratios)
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors='black', linestyles='solid')
    plt.xticks(list(range(number_of_samples, 0, -500)) + [0])
    plt.legend()
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks([mrs_iterations])
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_experiment_comparison_auc(auc_score_mrs, std_aucs_mrs, auc_score_experiment, std_aucs_experiment,
                                   experiment_label, drop, file_name, number_of_samples, save=False):
    aucs_upper = np.minimum(auc_score_mrs + std_aucs_mrs, 1)
    aucs_lower = np.maximum(auc_score_mrs - std_aucs_mrs, 0)

    aucs_upper_experiment = np.minimum(auc_score_experiment + std_aucs_experiment, 1)
    aucs_lower_experiment = np.maximum(auc_score_experiment - std_aucs_experiment, 0)

    x_labels = range(number_of_samples, drop, -drop)

    plt.fill_between(x_labels, aucs_lower, aucs_upper, color='blue', alpha=0.2)
    plt.plot(x_labels, auc_score_mrs, color='blue', linestyle='-', label='MRS')

    plt.fill_between(x_labels, aucs_lower_experiment, aucs_upper_experiment, color='orange', alpha=0.2)
    plt.plot(x_labels, auc_score_experiment, linestyle=':', color='orange', label=experiment_label)

    plt.plot(len(auc_score_mrs) * drop * [0.5], color='black', linestyle='--')

    plt.ylabel('AUROC')
    plt.xlabel('Number of remaining samples')
    plt.xticks(list(range(number_of_samples, 0, -100)) + [0])
    plt.legend()
    plt.gca().invert_xaxis()
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_experiment_comparison_mmd(median_mmd, std_mmd, median_mmd_experiment, std_mmd_experiment,
                                   experiment_label, drop, mmd_iteration, file_name, number_of_samples,
                                   save=False):
    mmd_upper = np.minimum(median_mmd + std_mmd, 1)
    mmd_lower = np.maximum(median_mmd - std_mmd, 0)

    mmd_upper_experiment = np.minimum(median_mmd_experiment + std_mmd_experiment, 1)
    mmd_lower_experiment = np.maximum(median_mmd_experiment - std_mmd_experiment, 0)

    x_labels = range(number_of_samples, drop * mmd_iteration, -drop * mmd_iteration)

    plt.fill_between(x_labels, mmd_lower, mmd_upper, color='blue', alpha=0.2)
    plt.plot(x_labels, median_mmd, color='blue', linestyle='-', label='MRS')

    plt.fill_between(x_labels, mmd_lower_experiment, mmd_upper_experiment, color='orange', alpha=0.2)
    plt.plot(x_labels, median_mmd_experiment, linestyle=':', color='orange', label=experiment_label)

    plt.ylabel('Maximum mean discrepancy')
    plt.xlabel('Number of remaining samples')
    plt.xticks(list(range(number_of_samples, 0, -100)) + [0])
    plt.legend()
    plt.gca().invert_xaxis()
    if save:
        plt.savefig(f'{file_name}.pdf')
    plt.show()


def plot_asams(weighted_asams, unweighted_asams, columns,  plot_directory):
    asams = np.append(weighted_asams.values, unweighted_asams.values)
    asams_difference = weighted_asams-unweighted_asams
    colors = ['g' if c >= 0 else 'r' for c in asams_difference]
    columns = np.repeat(columns, 2)
    sns.boxplot(
        x=columns,
        y=asams,
        palette=colors
    )
    plt.xticks(rotation=70)
    plt.savefig(f'{plot_directory}/asams.pdf', bbox_inches='tight')
    plt.clf()
