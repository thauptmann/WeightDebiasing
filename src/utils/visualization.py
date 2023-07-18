import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from cycler import cycler

sns.set_theme(style="ticks")


def plot_cumulative_distribution_function(
    N, R, file_name: str, weights_one, weights_two, method_one, method_two, wide=True
):
    if wide:
        plt.figure(figsize=(10, 5))
    plot_directory = file_name / "cumulative_distributions"
    plot_directory.mkdir(exist_ok=True)
    for column_name in N.columns:
        sns.ecdfplot(N, x=column_name, label="GBS")
        sns.ecdfplot(R, x=column_name, label="Allensbach", linestyle="dashed")
        sns.ecdfplot(
            N, x=column_name, weights=weights_one, label=method_one, linestyle="dotted"
        )
        sns.ecdfplot(
            N, x=column_name, weights=weights_two, label=method_two, linestyle="dashdot"
        )
        plt.legend()
        plt.savefig(plot_directory / f"{column_name}.pdf")
        plt.clf()


def plot_feature_histograms(
    N, R, file_name, bins, weights_one, weights_two, method_one, method_two
):
    plot_directory = file_name / "histograms"
    plot_directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(10, 5))
    for column_name in N.columns:
        sns.histplot(
            N, x=column_name, ax=ax[0], bins=bins, stat="probability", kde=True
        ).set_title("GBS")
        sns.histplot(
            R, x=column_name, ax=ax[1], bins=bins, stat="probability", kde=True
        ).set_title("Allensbach")
        sns.histplot(
            N,
            x=column_name,
            weights=weights_one,
            ax=ax[2],
            bins=bins,
            stat="probability",
            kde=True,
        ).set_title(method_one)
        sns.histplot(
            N,
            x=column_name,
            weights=weights_two,
            ax=ax[3],
            bins=bins,
            stat="probability", kde=True
        ).set_title(method_two)
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


def plot_statistical_analysis(
    bins: int,
    N: np.ndarray,
    R: np.ndarray,
    visualisation_path: Path,
    weights_one: list[float],
    weights_two: list[float],
    method_one: str = "",
    method_two: str = "",
):
    plot_cumulative_distribution_function(
        N, R, visualisation_path, weights_one, weights_two, method_one, method_two
    )
    plot_feature_histograms(
        N, R, visualisation_path, bins, weights_one, weights_two, method_one, method_two
    )


def plot_results_with_variance(
    metric_list: list[float], visualisation_path: Path, suffix: str = "", metric="MMD"
):
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


def mrs_progress_visualization(
    mmd_list: list[float],
    auc_list,
    relative_bias_list,
    mrs_iteration_list,
    drop,
    number_of_samples,
    save_path,
):
    plot_auc_average(
        np.squeeze(np.mean(auc_list, axis=0)),
        np.squeeze(np.std(auc_list, axis=0)),
        drop,
        save_path / "auc_mean_with_mrs_iterations",
        number_of_samples,
        mrs_iterations=mrs_iteration_list,
        wide=True,
    )
    plot_mmds_average(
        np.squeeze(np.mean(mmd_list, axis=0)),
        np.squeeze(np.std(mmd_list, axis=0)),
        drop,
        1,
        save_path / "mean_mmds",
        mrs_iteration_list,
        number_of_samples,
    )

    if relative_bias_list.size != 0:
        plot_relative_bias(
            np.mean(np.array(relative_bias_list), axis=0),
            np.std(np.array(relative_bias_list), axis=0),
            save_path / "Relative_Bias",
            mrs_iteration_list,
            number_of_samples,
            drop,
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
        plt.figure(figsize=(12.8, 4.8))

    aucs_upper = np.minimum(auc_score + std_aucs, 1)
    aucs_lower = np.maximum(auc_score - std_aucs, 0)
    stop = number_of_samples - ((auc_score.size) * drop)
    x_labels = list(range(number_of_samples, stop, -drop))
    plt.fill_between(x_labels, aucs_lower, aucs_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, auc_score, color="blue", linestyle="-")
    random_line = len(x_labels) * [0.5]
    plt.plot(
        x_labels,
        random_line,
        color="black",
        linestyle="--",
        label="Random",
    )
    plt.ylabel("AUROC")
    mrs_iterations = number_of_samples - (np.array(mrs_iterations))
    minimum = min(0.5, np.min(aucs_lower)) - 0.01
    maximum = plt.gca().get_ylim()[1] + 0.01
    plt.margins(0.05, 0)
    plt.vlines(
        mrs_iterations,
        minimum,
        maximum,
        colors="black",
        linestyles="solid",
    )
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [4]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of Remaining Samples")
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(list(mrs_iterations))
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_mmds_average(
    mmds, std, drop, mmd_iteration, file_name, mrs_iterations, number_of_samples
):
    mmds_upper = mmds + std
    mmds_lower = np.maximum(mmds - std, 0)
    stop = number_of_samples - ((mmds.size) * drop)
    x_labels = range(number_of_samples, stop, -(drop * mmd_iteration))
    plt.fill_between(x_labels, mmds_lower, mmds_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, mmds, linestyle="-")
    minimum = np.min(mmds_lower) - 0.001
    maximum = plt.gca().get_ylim()[1] + 0.001
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors="black", linestyles="solid")
    plt.ylabel("Maximum Mean Discrepancy")
    plt.xlabel("Number of Remaining Samples")
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [4]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(mrs_iterations)
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_experiment_comparison_auc(
    auc_score_mrs,
    std_aucs_mrs,
    auc_score_experiment,
    std_aucs_experiment,
    experiment_label,
    drop,
    file_name,
    number_of_samples,
):
    aucs_upper = np.minimum(auc_score_mrs + std_aucs_mrs, 1)
    aucs_lower = np.maximum(auc_score_mrs - std_aucs_mrs, 0)

    aucs_upper_experiment = np.minimum(auc_score_experiment + std_aucs_experiment, 1)
    aucs_lower_experiment = np.maximum(auc_score_experiment - std_aucs_experiment, 0)

    stop = number_of_samples - ((auc_score_mrs.size) * drop)
    x_labels = list(range(number_of_samples, stop, -drop))

    plt.fill_between(x_labels, aucs_lower, aucs_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, auc_score_mrs, color="blue", linestyle="-", label="MRS")

    plt.fill_between(
        x_labels,
        aucs_lower_experiment,
        aucs_upper_experiment,
        color="orange",
        alpha=0.2,
    )
    plt.plot(
        x_labels,
        auc_score_experiment,
        linestyle=":",
        color="orange",
        label=experiment_label,
    )

    plt.plot(len(auc_score_mrs) * drop * [0.5], color="black", linestyle="--")

    plt.ylabel("AUROC")
    plt.xlabel("Number of Remaining Samples")
    plt.xticks(list(range(number_of_samples, 0, -100)) + [4])
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f"{file_name}.pdf")


def plot_experiment_comparison_mmd(
    mean_mmd,
    std_mmd,
    mean_mmd_experiment,
    std_mmd_experiment,
    experiment_label,
    drop,
    mmd_iteration,
    file_name,
    number_of_samples,
):
    mmd_upper = np.minimum(mean_mmd + std_mmd, 1)
    mmd_lower = np.maximum(mean_mmd - std_mmd, 0)

    mmd_upper_experiment = np.minimum(mean_mmd_experiment + std_mmd_experiment, 1)
    mmd_lower_experiment = np.maximum(mean_mmd_experiment - std_mmd_experiment, 0)

    stop = number_of_samples - ((mean_mmd.size) * drop)
    x_labels = range(number_of_samples, stop, -(drop * mmd_iteration))

    plt.fill_between(x_labels, mmd_lower, mmd_upper, color="blue", alpha=0.2)
    plt.plot(x_labels, mean_mmd, color="blue", linestyle="-", label="MRS")

    plt.fill_between(
        x_labels, mmd_lower_experiment, mmd_upper_experiment, color="orange", alpha=0.2
    )
    plt.plot(
        x_labels,
        mean_mmd_experiment,
        linestyle=":",
        color="orange",
        label=experiment_label,
    )

    plt.ylabel("Maximum Mean Discrepancy")
    plt.xlabel("Number of Remaining Samples")
    plt.xticks(list(range(number_of_samples, 0, -100)) + [4])
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f"{file_name}.pdf")
    plt.close()


default_cycle = cycler(
    "linestyle",
    [
        ":",
        "-.",
        (0, (3, 5, 1, 5, 1, 5)),
        "-",
    ],
) + cycler(color=["blue", "orange", "orangered", "cyan"])


def plot_rocs(roc_list, file_name):
    plt.rc("")
    plt.rc("axes", prop_cycle=default_cycle)
    for fper, tper, std, deleted_elements in roc_list:
        tpfrs_higher = np.minimum(tper + std, 1)
        tpfrs_lower = np.maximum(tper - std, 0)
        plt.plot(fper, tper, label=f"{deleted_elements} samples removed")
        plt.fill_between(fper, tpfrs_lower, tpfrs_higher, alpha=0.2)
    plt.plot(
        [0, 1], [0, 1], color="black", linestyle="--", label="Random", linewidth=0.8
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"{file_name}.pdf")
    plt.close()


def plot_relative_bias(
    mean_relative_bias_list,
    std_relative_bias_list,
    file_name,
    mrs_iterations,
    number_of_samples,
    drop,
):
    plt.xlabel("Number of Remaining Samples")
    plt.ylabel("Relative Bias")
    ratio_upper = mean_relative_bias_list + std_relative_bias_list
    ratio_lower = (mean_relative_bias_list - std_relative_bias_list).clip(min=0)
    stop = number_of_samples - (mean_relative_bias_list.size * drop)
    x_labels = list(range(number_of_samples, stop, -drop))

    plt.plot(
        x_labels,
        mean_relative_bias_list,
        linestyle="-",
        color="blue",
    )
    plt.fill_between(x_labels, ratio_lower, ratio_upper, color="blue", alpha=0.2)
    minimum = plt.gca().get_ylim()[0]
    maximum = plt.gca().get_ylim()[1]
    plt.margins(0.05, 0)
    mrs_iterations = number_of_samples - np.array(mrs_iterations)
    plt.vlines(mrs_iterations, minimum, maximum, colors="black", linestyles="solid")
    x_ticks = list(range(number_of_samples, stop, -(number_of_samples // 5))) + [
        stop + drop
    ]
    plt.xticks(x_ticks)
    plt.gca().invert_xaxis()
    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    plt.xticks(mrs_iterations)
    [tick.set_color("blue") for tick in plt.gca().get_xticklabels()]
    plt.savefig(f"{file_name}.pdf")
    plt.close()
