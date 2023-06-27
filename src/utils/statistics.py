import numpy as np
import statsmodels.api as sm


def logistic_regression(allensbach_gbs, weights):
    y = allensbach_gbs["Wahlteilnahme"]
    X = allensbach_gbs["Resilienz"]
    X = sm.add_constant(X)
    model_gbs = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.Logit()))
    results_gbs = model_gbs.fit()
    lr_pvalue_gbs = results_gbs.pvalues[1]

    model_all = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.Logit()),
        freq_weights=weights,
    )
    results_weighted = model_all.fit()
    lr_pvalue_weighted = results_weighted.pvalues[1]

    return lr_pvalue_gbs, lr_pvalue_weighted


def write_result_dict(
    columns,
    weighted_mmds_list,
    biases_list,
    wasserstein_parameter_list,
    remaining_samples_list,
    mse_list,
    auroc_list,
    accuracy_list,
    precision_list,
    f_score_list,
    recall_list,
    tn_list,
    fn_list,
    tp_list,
    fp_list,
    number_of_samples,
):
    result_dict = {
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
        "auroc": {
            "mean": np.nanmean(auroc_list),
            "sd": np.nanstd(auroc_list),
        },
        "accuracy": {
            "mean": np.nanmean(accuracy_list),
            "sd": np.nanstd(accuracy_list),
        },
        "precision": {
            "mean": np.nanmean(precision_list),
            "sd": np.nanstd(precision_list),
        },
        "auprc": {
            "mean": np.nanmean(f_score_list),
            "sd": np.nanstd(f_score_list),
        },
        "recall": {
            "mean": np.nanmean(recall_list),
            "sd": np.nanstd(recall_list),
        },
        "true negative": {
            "mean": np.nanmean(tn_list),
            "sd": np.nanstd(tn_list),
        },
        "false negative": {
            "mean": np.nanmean(fn_list),
            "sd": np.nanstd(fn_list),
        },
        "true positive": {
            "mean": np.nanmean(tp_list),
            "sd": np.nanstd(tp_list),
        },
        "false positive": {
            "mean": np.nanmean(fp_list),
            "sd": np.nanstd(fp_list),
        },
        "mse": {
            "mean": np.nanmean(mse_list),
            "sd": np.nanstd(mse_list),
        },
        "remaining samples": {
            "mean": np.nanmean(remaining_samples_list),
            "sd": np.nanstd(remaining_samples_list),
        },
        "all_samples": number_of_samples,
    }

    mean_biases = np.nanmean(biases_list, axis=0)
    sd_biases = np.nanstd(biases_list, axis=0)
    mean_wasserstein = np.nanmean(wasserstein_parameter_list, axis=0)
    sd_wasserstein = np.nanstd(wasserstein_parameter_list, axis=0)
    for index, column in enumerate(columns):
        result_dict[f"{column}_relative_bias"] = {
            "bias mean": mean_biases[index],
            "bias sd": sd_biases[index],
            "wasserstein mean": mean_wasserstein[index],
            "wasserstein sd": sd_wasserstein[index],
        }

    return result_dict
