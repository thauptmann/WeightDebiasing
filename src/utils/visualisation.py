import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_line(auc_score, file_name, title="", plot_random_line=False):
    plt.plot(auc_score, color="blue", linestyle="-", label=title)
    if plot_random_line:
        plt.plot(len(auc_score) * [0.5], color="black", linestyle="--", label="Random")
    plt.ylabel(title)
    plt.xlabel("Iteration")
    plt.savefig(file_name / f"{title}.pdf")
    plt.close()


def plot_feature_distribution(df, columns, file_name, weights):
    plot_directory = file_name / "cumulative_distributions"
    plot_directory.mkdir(exist_ok=True)
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for column_name in columns:
        sns.ecdfplot(N, x=column_name, label="Non-Representative")
        sns.ecdfplot(R, x=column_name, label="Representative")
        sns.ecdfplot(N, x=column_name, weights=weights, label="Weighted")
        plt.legend(title="Data set")
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.close()


def plot_feature_histograms(df, columns, file_name, bins, weights):
    plot_directory = file_name / "histograms"
    plot_directory.mkdir(exist_ok=True)
    N = df[df["label"] == 1]
    R = df[df["label"] == 0]
    for column_name in columns:
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


def plot_rocs(start_roc, weighted_roc, file_name):
    for fper, tper, std, deleted_elements, name in zip(
        (start_roc, weighted_roc), ("Start", "Weighted")
    ):
        tpfrs_higher = np.minimum(tper + std, 1)
        tpfrs_lower = np.maximum(tper - std, 0)
        plt.plot(fper, tper, label=f"{name}")
        plt.fill_between(fper, tpfrs_lower, tpfrs_higher, alpha=0.2)
    plt.plot(
        [0, 1], [0, 1], color="black", linestyle="--", label="Random", linewidth=0.8
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{file_name}.pdf")
    plt.clf()


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
    plt.axvline(same_weights_positition, color='k')
    plt.savefig(f"{path}/weights_{iteration}.pdf", bbox_inches="tight")
    plt.clf()


def plot_probabilities(probabilities, path, iteration, bins=50):
    sns.histplot(x=probabilities, bins=bins)
    plt.savefig(f"{path}/probabilities_{iteration}.pdf", bbox_inches="tight")
    plt.clf()
