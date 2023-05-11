import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")


def plot_cumulative_distribution(N, R, file_name, weights):
    plot_directory = file_name / "cumulative_distributions"
    plot_directory.mkdir(exist_ok=True)
    for column_name in N.columns:
        sns.ecdfplot(N, x=column_name, label="Non-Representative")
        sns.ecdfplot(R, x=column_name, label="Representative", linestyle="dashed")
        sns.ecdfplot(
            N, x=column_name, weights=weights, label="Weighted", linestyle="dotted"
        )
        plt.legend(title="Data set")
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.clf()


def plot_feature_histograms(N, R, file_name, bins, weights):
    plot_directory = file_name / "histograms"
    plot_directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    for column_name in N.columns:
        sns.histplot(
            N, x=column_name, ax=ax[0], bins=bins, stat="probability"
        ).set_title("Non-Representative")
        sns.histplot(
            R, x=column_name, ax=ax[1], bins=bins, stat="probability"
        ).set_title("Representative")
        sns.histplot(
            N, x=column_name, weights=weights, ax=ax[2], bins=bins, stat="probability"
        ).set_title("Weighted")
        fig.savefig(plot_directory / f"{column_name}.pdf")
        [axis.clear() for axis in ax]
    plt.clf()


def plot_weights(weights, path, iteration, title="", bins=25):
    path.mkdir(exist_ok=True)
    weights = weights / sum(weights)
    sns.histplot(x=weights, bins=bins).set_title(title)
    same_weights_positition = 1 / len(weights)
    plt.axvline(same_weights_positition, color="k")
    plt.savefig(f"{path}/weights_{iteration}.pdf", bbox_inches="tight")
    plt.clf()


def plot_ratio(values, representative_ratio, title, path):
    beautified_title = title.replace("_", " ")
    plt.plot(values, color="blue", linestyle="-", label=beautified_title)
    plt.axhline(y=representative_ratio, color="k", linestyle="--")
    plt.ylabel(beautified_title)
    plt.xlabel("Iteration")
    plt.savefig(Path(path) / f"{title}.pdf")
    plt.clf()


def plot_gbs_results(
    bins: int,
    N: np.ndarray,
    R: np.ndarray,
    visualisation_path: Path,
    weights: list[float],
):
    plot_cumulative_distribution(N, R, visualisation_path, weights)
    plot_feature_histograms(N, R, visualisation_path, bins, weights)
    plot_weights(weights / sum(weights), visualisation_path, 0, bins)


def plot_results_with_variance(
    ratio_list: list[float],
    mmd_list: list[float],
    representative_ratio: float,
    visualisation_path: Path,
    suffix: str = "",
    plot_mean: bool = False,
):
    plot_metric_with_variance(mmd_list, visualisation_path, suffix, "MMD")
    if plot_mean:
        plot_mean_with_variance(
            ratio_list, representative_ratio, visualisation_path, suffix
        )


def plot_metric_with_variance(metric_list, visualisation_path, suffix, metric):
    mean_metric = np.nanmean(metric_list, axis=0)
    sd_metric = np.nanstd(metric_list, axis=0)
    plt.plot(range(len(mean_metric)), mean_metric, color="blue")
    plt.fill_between(
        x=range(len(mean_metric)),
        y1=mean_metric - sd_metric,
        y2=mean_metric + sd_metric,
        color="blue",
        alpha=0.5,
    )
    plt.ylabel(f"Weighted {metric}")
    plt.xlabel("Pass")
    plt.savefig(Path(visualisation_path) / f"weighted_{metric}_{suffix}.pdf")
    plt.clf()


def plot_mean_with_variance(mean_list, representative_mean, visualisation_path, suffix):
    mean_mean = np.nanmean(mean_list, axis=0)
    sd_mean = np.nanstd(mean_list, axis=0)
    plt.plot(
        range(len(mean_mean)),
        mean_mean,
        color="blue",
        label="Weighted Mean",
        linestyle="--",
    )
    plt.plot(
        (len(mean_mean)) * [representative_mean],
        color="black",
        linestyle="-",
        label="Population Mean",
    )
    plt.fill_between(
        x=range(len(mean_mean)),
        y1=mean_mean - sd_mean,
        y2=mean_mean + sd_mean,
        color="blue",
        alpha=0.5,
    )
    plt.ylabel("Mean")
    plt.xlabel("Pass")
    plt.legend()
    plt.savefig(Path(visualisation_path) / f"weighted_mean_{suffix}.pdf")
    plt.clf()


def mrs_progress_visualization(
    mmd_list, auc_list, drop, number_of_iterations, number_of_samples, save_path
):
    plot_auc_average(
        auc_list,
        np.std(auc_list, axis=0),
        drop,
        save_path + "gesis_auc_mean_with_iterations",
        number_of_samples,
        mrs_iterations=number_of_iterations,
        wide=True,
    )
    # plot_rocs(gesis_mean_rocs, file_directory+"gesis_mean_rocs", save=save)
    plot_mmds_average(
        mmd_list,
        np.std(mmd_list, axis=0),
        drop,
        1,
        save_path + "mean_mmds_gesis",
        number_of_iterations,
        number_of_samples,
    )


def plot_auc_average(
    auc_score,
    std_aucs,
    drop,
    file_name,
    number_of_samples,
    mrs_iterations,
    wide=True,
):
    if wide:
        plt.figure(figsize=(6.4 * 2, 4.8))
    aucs_upper = np.minimum(auc_score + std_aucs, 1)
    aucs_younger = np.maximum(auc_score - std_aucs, 0)
    x_labels = range(number_of_samples, drop, -drop)
    plt.fill_between(x_labels, aucs_younger, aucs_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, auc_score, color="blue", linestyle="-")
    plt.plot(
        (len(x_labels) - 1) * drop * [0.5],
        color="black",
        linestyle="--",
        label="Random",
    )
    plt.ylabel("AUROC")
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    minimum = min(0.5, np.min(aucs_younger))
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    plt.vlines(mrs_iterations, minimum, maximum, colors="black", linestyles="solid")
    plt.xticks(list(range(number_of_samples, 0, -100)) + [0])
    plt.gca().invert_xaxis()
    plt.xlabel("Number of remaining samples")
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(list(mrs_iterations))
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")


def plot_mmds_average(mmds, std, drop, mmd_iteration, file_name, mrs_iterations,
                      number_of_samples):
    mmds_upper = np.minimum(mmds + std, 1)
    mmds_more_negative = np.maximum(mmds - std, 0)
    x_labels = range(number_of_samples, drop*mmd_iteration, -drop*mmd_iteration)
    plt.fill_between(x_labels, mmds_more_negative, mmds_upper, color='black', alpha=0.2)
    plt.plot(x_labels, mmds, linestyle='-')
    minimum = np.min(mmds_more_negative)
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
    plt.savefig(f'{file_name}.pdf')