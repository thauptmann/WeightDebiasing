import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")


def plot_line(values, path, title="", plot_random_line=False):
    beautified_title = title.replace("_", " ")
    plt.plot(values, color="blue", linestyle="-", label=beautified_title)
    if plot_random_line:
        plt.plot(len(values) * [0.5], color="black", linestyle="--", label="Random")
    plt.ylabel(beautified_title)
    plt.xlabel("Iteration")
    plt.savefig(Path(path) / f"{title}.pdf")
    plt.clf()


def plot_feature_distribution(N, R, file_name, weights):
    plot_directory = file_name / "cumulative_distributions"
    plot_directory.mkdir(exist_ok=True)
    for column_name in N.columns:
        sns.ecdfplot(N, x=column_name, label="Non-Representative")
        sns.ecdfplot(R, x=column_name, label="Representative", linestyle="dashed")
        sns.ecdfplot(N, x=column_name, weights=weights, label="Weighted", linestyle="dotted")
        plt.legend(title="Data set")
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.clf()


def plot_feature_histograms(N, R, file_name, bins, weights):
    plot_directory = file_name / "histograms"
    plot_directory.mkdir(exist_ok=True)
    for column_name in N.columns:
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
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
        plt.close()


def plot_asams(weighted_asams, asams, columns, plot_directory):
    asams_difference = weighted_asams.values - asams.values
    interleave_list = [None] * (len(asams) + len(weighted_asams))
    interleave_list[::2] = asams
    interleave_list[1::2] = weighted_asams

    mean_weighted_asams = np.mean(weighted_asams)
    mean_asams = np.mean(asams)
    mean_differences = mean_weighted_asams - mean_asams
    interleave_list = np.append(interleave_list, [mean_asams, mean_weighted_asams])
    asams_difference = np.append(asams_difference, mean_differences)
    colors = ["g" if c >= 0 else "r" for c in asams_difference]
    columns = np.repeat(columns, 2)
    columns = np.append(columns, ["Mean ASAM", "Mean ASAM"])
    sns.boxplot(x=columns, y=interleave_list, palette=colors)
    plt.xticks(rotation=90)
    plt.savefig(f"{plot_directory}/asams.pdf", bbox_inches="tight")
    plt.clf()


def plot_weights(weights, path, iteration, bins=50):
    sns.histplot(x=weights, bins=bins)
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


def plot_results(
    asams,
    asams_values,
    bins,
    columns,
    mmd,
    N,
    R,
    visualisation_path,
    weighted_asams,
    weighted_mmd,
    weights,
):
    plot_asams(weighted_asams, asams_values, columns, visualisation_path)
    plot_feature_distribution(N, R, visualisation_path, weights)
    plot_feature_histograms(N, R, visualisation_path, bins, weights)
    plot_line(asams, visualisation_path, title="ASAM")
    plot_line([mmd, weighted_mmd], visualisation_path, title="MMD")
    plot_weights(weights / sum(weights), visualisation_path, 0, bins)
