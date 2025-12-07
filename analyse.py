import statsmodels.api as sm 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def _encode_hot_ones(lf, cols):
    categorial_cols = [c for c, dt in zip(lf.columns, lf.dtypes) if c in cols and dt in (pl.Utf8, pl.Categorical, pl.Enum)]
    if categorial_cols:
        return lf.collect().to_dummies(columns=categorial_cols)
    return lf.collect()

def unajusted_regression(treatment, outcome):
    X = sm.add_constant(treatment)

    model = sm.OLS(outcome, X).fit(cov_type="HC3")

    return{
        "tau": model.params[1],
        "p_value": model.pvalues[1],
        "ci_low": model.conf_int(alpha=0.05)[1][0],
        "ci_high": model.conf_int(alpha=0.05)[1][1]
    }


def ajusted_regression(treatment, outcome, confounders):
    X = sm.add_constant(np.column_stack((treatment, confounders)))

    model = sm.OLS(outcome, X).fit(cov_type="HC3")

    return{
        "tau": model.params[1],
        "p_value": model.pvalues[1],
        "ci_low": model.conf_int(alpha=0.05)[1][0],
        "ci_high": model.conf_int(alpha=0.05)[1][1]
    }

def propensity_score_ipw_ate(treatment, outcome, confounders):
    ps_model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000))
    ])
    ps_model.fit(confounders, treatment)
    ps = ps_model.predict_proba(confounders)[:, 1]
    ps = np.clip(ps, .01, .99)
    weight = np.where(treatment == 1, 1/ps, 1/(1-ps))

    x = sm.add_constant(treatment)
    wls = sm.WLS(outcome, x, weights=weight).fit(cov_type="HC3")
    return {
        "ate": wls.params[1],
        "p_value" : wls.pvalues[1],
        "ci_low": wls.conf_int(alpha=0.05)[1][0],
        "ci_high": wls.conf_int(alpha=0.05)[1][1]
    }

def kaplan_meier(pd_df):
    exposed = pd_df[pd_df["exposed_in_24h"] == 1]
    control = pd_df[pd_df["exposed_in_24h"] == 0]

    kaplan_meier_exposed = KaplanMeierFitter()
    kaplan_meier_control = KaplanMeierFitter()

    kaplan_meier_exposed.fit(exposed["time_to_follow_up_event_rel_to_inclusion"], exposed["cumulative_outcome"], label="Exposed")
    kaplan_meier_control.fit(control["time_to_follow_up_event_rel_to_inclusion"], control["cumulative_outcome"], label="Control")

    ax = kaplan_meier_exposed.plot_survival_function()
    kaplan_meier_control.plot_survival_function(ax=ax)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")

    # Log-rank test (difference between curves)
    res = logrank_test(
        exposed["time_to_follow_up_event_rel_to_inclusion"], control["time_to_follow_up_event_rel_to_inclusion"],
        event_observed_A=exposed["cumulative_outcome"],
        event_observed_B=control["cumulative_outcome"]
    )

    return {
        "exposed_medium_survival" : kaplan_meier_exposed.median_survival_time_,
        "control_medium_survival" : kaplan_meier_control.median_survival_time_,
        "log_rank_p_value": res.p_value,
        "figure": ax.get_figure()
    }