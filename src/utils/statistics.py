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
    columns,
    weighted_mmds_list,
    biases_list,
    wasserstein_parameter_list,
    remaining_samples_list,
    mse_list,
    tree_auroc_list,
    tree_accuracy_list,
    tree_precision_list,
    tree_f_score_list,
    tree_recall_list,
    tree_tn_list,
    tree_fn_list,
    tree_tp_list,
    tree_fp_list,
    svc_auroc_list,
    svc_accuracy_list,
    svc_precision_list,
    svc_f_score_list,
    svc_recall_list,
    svc_tn_list,
    svc_fn_list,
    svc_tp_list,
    svc_fp_list,
    number_of_samples,
):
    result_dict = {
        "MMDs": {
            "mean": np.nanmean(weighted_mmds_list),
            "sd": np.nanstd(weighted_mmds_list),
        },
        "tree auroc": {
            "mean": np.nanmean(tree_auroc_list),
            "sd": np.nanstd(tree_auroc_list),
        },
        "tree accuracy": {
            "mean": np.nanmean(tree_accuracy_list),
            "sd": np.nanstd(tree_accuracy_list),
        },
        "tree precision": {
            "mean": np.nanmean(tree_precision_list),
            "sd": np.nanstd(tree_precision_list),
        },
        "tree f score": {
            "mean": np.nanmean(tree_f_score_list),
            "sd": np.nanstd(tree_f_score_list),
        },
        "tree recall": {
            "mean": np.nanmean(tree_recall_list),
            "sd": np.nanstd(tree_recall_list),
        },
        "tree true negative": {
            "mean": np.nanmean(tree_tn_list),
            "sd": np.nanstd(tree_tn_list),
        },
        "tree false negative": {
            "mean": np.nanmean(tree_fn_list),
            "sd": np.nanstd(tree_fn_list),
        },
        "tree true positive": {
            "mean": np.nanmean(tree_tp_list),
            "sd": np.nanstd(tree_tp_list),
        },
        "tree false positive": {
            "mean": np.nanmean(tree_fp_list),
            "sd": np.nanstd(tree_fp_list),
        },
        "svc auroc": {
            "mean": np.nanmean(svc_auroc_list),
            "sd": np.nanstd(svc_auroc_list),
        },
        "svc accuracy": {
            "mean": np.nanmean(svc_accuracy_list),
            "sd": np.nanstd(svc_accuracy_list),
        },
        "svc precision": {
            "mean": np.nanmean(svc_precision_list),
            "sd": np.nanstd(svc_precision_list),
        },
        "svc f score": {
            "mean": np.nanmean(svc_f_score_list),
            "sd": np.nanstd(svc_f_score_list),
        },
        "svc recall": {
            "mean": np.nanmean(svc_recall_list),
            "sd": np.nanstd(svc_recall_list),
        },
        "svc true negative": {
            "mean": np.nanmean(svc_tn_list),
            "sd": np.nanstd(svc_tn_list),
        },
        "svc false negative": {
            "mean": np.nanmean(svc_fn_list),
            "sd": np.nanstd(svc_fn_list),
        },
        "svc true positive": {
            "mean": np.nanmean(svc_tp_list),
            "sd": np.nanstd(svc_tp_list),
        },
        "svc false positive": {
            "mean": np.nanmean(svc_fp_list),
            "sd": np.nanstd(svc_fp_list),
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
