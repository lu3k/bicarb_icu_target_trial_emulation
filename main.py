import analysis_df
import analyse

exposure = analysis_df.get_exposure_np()
outcome = analysis_df.get_outcome_np()
confounders = analysis_df.get_confounders_np(["Admission Age (years)"])

unajusted_regression = analyse.unajusted_regression(exposure, outcome)
ajusted_regression = analyse.ajusted_regression(exposure, outcome, confounders)
prosperity = analyse.propensity_score_ipw_ate(exposure, outcome, confounders)

km = analysis_df.get_kaplan_meier_pd()
km_results = analyse.kaplan_meier(km)

print(f"Total patients : {len(exposure)}, exposed patients : {len(exposure[exposure == 1])}")

print(f"Unajusted Regression : {unajusted_regression["tau"]}, p={unajusted_regression["p_value"]}, CI95=[{unajusted_regression["ci_low"]}, {unajusted_regression["ci_high"]}]")
print(f"Ajusted Regression : {ajusted_regression["tau"]}, p={ajusted_regression["p_value"]}, CI95=[{ajusted_regression["ci_low"]}, {ajusted_regression["ci_high"]}]")
print(f"Prosperity IPW : ATE={prosperity["ate"]}, p={prosperity["p_value"]}, CI95=[{prosperity["ci_low"]}, {prosperity["ci_high"]}]")
print(f"Kaplan Meier : medium survival (exposed) = {km_results["exposed_medium_survival"]}, medium survival (control) = {km_results["control_medium_survival"]}, p(LogRank)={km_results["log_rank_p_value"]}")

km_results["figure"].savefig("Kaplan_Meier.png")