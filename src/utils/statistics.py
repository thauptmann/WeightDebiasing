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
