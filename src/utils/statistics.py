import statsmodels.api as sm
from patsy import dmatrices
import scipy.stats as stats


def logistic_regression(allensbach_gbs, weights):
    y, X = dmatrices(
        "Wahlteilnahme ~ Resilienz", data=allensbach_gbs, return_type="dataframe"
    )
    model_gbs = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.logit()))
    results_gbs = model_gbs.fit()
    restricted_model_gbs = sm.GLM(
        y,
        X.drop(columns="Resilienz"),
        family=sm.families.Binomial(sm.families.links.logit()),
    )
    restricted_results_gbs = restricted_model_gbs.fit()

    model_all = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(sm.families.links.logit()),
        freq_weights=weights,
    )
    results_weighted = model_all.fit()
    restricted_model_all = sm.GLM(
        y,
        X.drop(columns="Resilienz"),
        family=sm.families.Binomial(sm.families.links.logit()),
        freq_weights=weights,
    )
    restricted_results_all = restricted_model_all.fit()

    results_gbs.summary()
    results_weighted.summary()

    loglikelihood_full = results_gbs.llf
    loglikelihood_restr = restricted_results_gbs.llf
    lrstat = -2 * (loglikelihood_restr - loglikelihood_full)
    lr_pvalue_gbs = stats.chi2.sf(lrstat, df=1)

    loglikelihood_full = results_weighted.llf
    loglikelihood_restr = restricted_results_all.llf
    lrstat = -2 * (loglikelihood_restr - loglikelihood_full)
    lr_pvalue_weighted = stats.chi2.sf(lrstat, df=1)

    print(f"{lr_pvalue_gbs=}")
    print(f"{lr_pvalue_weighted=}")
