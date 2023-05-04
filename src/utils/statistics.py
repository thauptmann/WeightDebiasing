import numpy as np
import statsmodels.api as sm


def logistic_regression(allensbach_gbs, weights):
    y = allensbach_gbs["Wahlteilnahme"]
    X = allensbach_gbs["Resilienz"]
    X = sm.add_constant(X)
    model_gbs = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.logit()))
    results_gbs = model_gbs.fit()
    lr_pvalue_gbs = results_gbs.pvalues[1]

    model_all = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.logit()),
        freq_weights=weights,
    )
    results_weighted = model_all.fit()
    lr_pvalue_weighted = results_weighted.pvalues[1]

    return lr_pvalue_gbs, lr_pvalue_weighted


def write_result_dict(
    weighted_mmds_list,
    dataset_ssmd_list,
    biases_list,
    parameter_ssmd_list,
    wasserstein_parameter_list,
    remaining_samples_list,
    auroc_list,
    error_rate_list,
    precision_list,
    recall_list,
    runtime_list,
    mse_list=None,
    scaled_N=None,
):
    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)

    mean_ssmds = np.nanmean(parameter_ssmd_list, axis=0)
    sd_ssmds = np.nanstd(parameter_ssmd_list, axis=0)

    mean_wasserstein = np.nanmean(wasserstein_parameter_list, axis=0)
    sd_wasserstein = np.nanstd(wasserstein_parameter_list, axis=0)

    result_dict = {
        "SSMD": {
            "mean": np.nanmean(dataset_ssmd_list),
            "sd": np.nanstd(dataset_ssmd_list),
        },
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
        "auroc": {
            "mean": np.nanmean(auroc_list),
            "sd": np.nanstd(auroc_list),
        },
        "accuracy": {
            "mean": np.nanmean(error_rate_list),
            "sd": np.nanstd(error_rate_list),
        },
        "precision": {
            "mean": np.nanmean(precision_list),
            "sd": np.nanstd(precision_list),
        },
        "recall": {
            "mean": np.nanmean(recall_list),
            "sd": np.nanstd(recall_list),
        },
        "runtime": {
            "mean": np.nanmean(runtime_list),
            "sd": np.nanstd(runtime_list),
        },
        "mse": {
            "mean": np.nanmean(mse_list),
            "sd": np.nanstd(mse_list),
        },
        "remaining samples": {
            "mean": np.nanmean(remaining_samples_list),
            "sd": np.nanstd(remaining_samples_list),
        },
        "all_samples": len(scaled_N),
    }

    for index, column in enumerate(scaled_N.drop(["label"], axis="columns").columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
            "ssmd mean": mean_ssmds[index],
            "ssmd sd": sd_ssmds[index],
            "wasserstein mean": mean_wasserstein[index],
            "wasserstein sd": sd_wasserstein[index],
        }

    return result_dict
