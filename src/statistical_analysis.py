import json
from pathlib import Path
from utils.command_line_arguments import (
    get_weighting_function,
    parse_command_line_arguments_statistical_analysis,
)
from utils.data_loader import load_dataset
from utils.metrics import calculate_rbf_gamma, compute_metrics, scale_df
from utils.statistics import logistic_regression
from utils.visualization import plot_statistical_analysis


bins = 25


def statistical_analysis(
    method_one,
    method_two,
    **args,
):
    """_summary_

    :param method_one: _description_
    :param method_two: _description_
    """
    file_directory = Path(__file__).parent
    result_path = Path(file_directory, "../results")
    visualisation_path = (
        result_path / "statistical analysis" / f"{method_one}_{method_two}"
    )
    visualisation_path.mkdir(exist_ok=True, parents=True)
    df, scale_columns, _ = load_dataset("gbs_allensbach")
    scaled_df, scaler = scale_df(df, scale_columns)

    weighting_function_one = get_weighting_function(method_one)
    weighting_function_two = get_weighting_function(method_two)

    gamma = calculate_rbf_gamma(scaled_df[scale_columns])

    scaled_N = scaled_df[scaled_df["label"] == 1]
    scaled_R = scaled_df[scaled_df["label"] == 0]

    weights_one = weighting_function_one(
        scaled_N,
        scaled_R,
        scale_columns,
        save_path=visualisation_path,
        drop=1,
        early_stopping=True,
    )

    weights_two = weighting_function_two(
        scaled_N,
        scaled_R,
        scale_columns,
        save_path=visualisation_path,
        drop=1,
        early_stopping=True,
    )

    (
        _,
        _,
        wasserstein_distances_one,
    ) = compute_metrics(
        scaled_N.copy(),
        scaled_R.copy(),
        weights_one,
        scaler,
        scale_columns,
        scale_columns,
        gamma,
    )

    (
        _,
        _,
        wasserstein_distances_two,
    ) = compute_metrics(
        scaled_N, scaled_R, weights_two, scaler, scale_columns, scale_columns, gamma
    )

    result_dict_one = {}
    for index, column in enumerate(scale_columns):
        result_dict_one[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances_one[index],
        }

    result_dict_two = {}
    for index, column in enumerate(scale_columns):
        result_dict_two[f"{column}_relative_bias"] = {
            "wasserstein": wasserstein_distances_two[index],
        }

    lr_pvalue_gbs, lr_pvalue_weighted_one = logistic_regression(
        scaled_N[scale_columns + ["Wahlteilnahme"]], weights_one
    )

    _, lr_pvalue_weighted_two = logistic_regression(
        scaled_N[scale_columns + ["Wahlteilnahme"]], weights_two
    )

    result_dict = {
        "logistig regression p values": {
            "GBS": lr_pvalue_gbs,
            f"Weighted {method_one}": lr_pvalue_weighted_one,
            f"Weighted {method_two}": lr_pvalue_weighted_two,
        }
    }

    with open(visualisation_path / "results.json", "w") as result_file:
        result_file.write(json.dumps(result_dict))

    with open(visualisation_path / f"results_{method_one}.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_one))

    with open(visualisation_path / f"results_{method_two}.json", "w") as result_file:
        result_file.write(json.dumps(result_dict_two))

    plot_statistical_analysis(
        bins,
        scaled_N[scale_columns],
        scaled_R[scale_columns],
        visualisation_path,
        weights_one,
        weights_two,
        method_one,
        method_two,
    )


if __name__ == "__main__":
    args = parse_command_line_arguments_statistical_analysis()
    statistical_analysis(args.method_one, args.method_two)
