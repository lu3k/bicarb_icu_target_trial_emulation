import polars as pl
from pathlib import Path
import sofa_helper
import statsmodels.api as sm 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data():
    print(f"[Cohort] Loading Cohort data ...")

    path = Path("bicarb_data")

    patient_information = pl.scan_parquet(path / "patient_information.parquet")
    medications = pl.scan_parquet(path / "medications.parquet")
    diagnoses = pl.scan_parquet(path / "diagnoses.parquet")
    procedures = pl.scan_parquet(path / "procedures.parquet")
    microbiology = pl.scan_parquet(path / "microbiology.parquet")
    ts_labs = pl.scan_parquet(path / "ts_labs.parquet")
    ts_vitals = pl.scan_parquet(path / "ts_vitals.parquet")
    ts_respiratory = pl.scan_parquet(path / "ts_respiratory.parquet")
    ts_intake_output = pl.scan_parquet(path / "ts_intake_output.parquet")

    return patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output

# Inclusion time is first severe acidosis time 
def get_inclusion_time(ts_labs):
    first_acidosis_time = ts_labs.filter(
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2) 
    ).group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("First Acidosis Time"))
    
    return first_acidosis_time

def get_bicarb_exposure(medications):
    bicarbs = medications.filter(
        pl.col("Drug Name").str.to_lowercase().str.contains("bicarb") |
        pl.col("Drug Name").str.to_lowercase().str.contains("hco") |
        pl.col("Drug Ingredient").str.contains("sodium bicarbonate")
    )
    return bicarbs

def get_bicarb_exposure_control_24h_after_inclusion(medication, ts_labs):
    inclusion_time = get_inclusion_time(ts_labs)
    #print(f"{inclusion_time.select("Global ICU Stay ID").unique().collect().to_series().len()} inclusion time.")
    bicarb_exposure = get_bicarb_exposure(medication)

    included_with_exposure = inclusion_time.join(
        bicarb_exposure.select(pl.col("Global ICU Stay ID"), pl.col("Drug Start Relative to Admission (seconds)")),
        on="Global ICU Stay ID",
        how="inner"
    )
    control_without_exposure = inclusion_time.join(
        bicarb_exposure.select(pl.col("Global ICU Stay ID")),
        on="Global ICU Stay ID",
        how="anti"
    )
    #print(f"{included_with_exposure.select("Global ICU Stay ID").unique().collect().to_series().len()} patients in included.")

    short_ids = (
        included_with_exposure
        .select([
            pl.col("Global ICU Stay ID"),
            (pl.col("Drug Start Relative to Admission (seconds)") - pl.col("First Acidosis Time")).alias("exp")
        ])
        .group_by("Global ICU Stay ID")
        .agg(
            (pl.col("exp") <= 24*3600).any().alias("keep")
        )
        .filter(pl.col("keep"))
        .select("Global ICU Stay ID")
    )
    exposure_in_24h = included_with_exposure.join(short_ids, on="Global ICU Stay ID", how="semi")
    control_after_24h = included_with_exposure.join(short_ids, on="Global ICU Stay ID", how="anti")

    #print(f"{bicarb_exposure.select("Global ICU Stay ID").unique().collect().to_series().len()} patients exposed to bicarb.")
    #print(f"{exposure_in_24h.select("Global ICU Stay ID").unique().collect().to_series().len()} patients exposed to bicarb in 24h.")
    #print(f"{control_after_24h.select("Global ICU Stay ID").unique().collect().to_series().len()} patients exposed to bicarb after 24h.")
    #print(f"{control_without_exposure.select("Global ICU Stay ID").unique().collect().to_series().len()} patients not exposed to bicarb.")
 
    return exposure_in_24h.select("Global ICU Stay ID"), pl.concat([control_after_24h.select("Global ICU Stay ID"), control_without_exposure.select("Global ICU Stay ID")])

# All cause mortality 
def get_all_cause_mortality(patient_information, ts_labs):
    print(patient_information.select(pl.col("Mortality in Hospital"), pl.col("Mortality in ICU"), pl.col("Mortality After ICU Discharge (days)"), pl.col("Hospital Length of Stay (days)"), pl.col("ICU Length of Stay (days)")).collect())
    print(patient_information.columns)

    # Time relative to first ICU admission
    first_time_acidosis = get_inclusion_time(ts_labs) # TODO: optimize so that doesnt have to recalculate




# Assessing organ failure at day 7 and mortality at day 28 or discharge (whichever comes sooner)
def follow_up(patient_info, ts_labs) :
    # Time relative to first ICU admission
    first_time_acidosis = get_inclusion_time(ts_labs) # TODO: optimize so that doesnt have to recalculate

    # TODO : ADD ORGANFAILIURE

    follow_up_lf = patient_info.join(
        first_time_acidosis, on="Global ICU Stay ID", how="inner"
    ).with_columns(
        # Death time or discharge time relative to inclusion time (since death is also the end of hosptialisation):
        # (Pre-ICU Length of Stay + time_to_inclusion) = Inclusion time relative to hospitalisation start
        # Hospital Length of stay - inclusion rel to hospitalisation start = time to death relative to inclusion 
        (pl.when(pl.col("Mortality in Hospital")).then(pl.lit("death")).otherwise(pl.lit("discharge"))).alias("follow_up_event"),
        (pl.col("Hospital Length of Stay (days)") - pl.col("Pre-ICU Length of Stay (days)") - pl.col("First Acidosis Time")/(3600*24)).alias("time_to_follow_up_event_rel_to_inclusion")
    )

    print(f"{follow_up_lf.filter((pl.col("follow_up_event") == "death")).select("Global ICU Stay ID").unique().collect().to_series().len()} patients died during hospitalisation. ")
    print(f"{follow_up_lf.filter((pl.col("follow_up_event") == "death") & (pl.col("time_to_follow_up_event_rel_to_inclusion") <= 28)).select("Global ICU Stay ID").unique().collect().to_series().len()} patients died in hospital in 28 days of inclusion. ")
    print(f"{follow_up_lf.filter((pl.col("follow_up_event") == "discharge")).select("Global ICU Stay ID").unique().collect().to_series().len()} patients got discharged. ")
    print(f"{follow_up_lf.filter((pl.col("follow_up_event") == "discharge") & (pl.col("time_to_follow_up_event_rel_to_inclusion") <= 28)).select("Global ICU Stay ID").unique().collect().to_series().len()} patients got discharged in 28 days of inclusion. ")


# All-cause, In-hospital mortality by day 28 and increase of 2 points (or to 4 points) in any SOFA category compared to baseline 
def outcome(patient_info, meds, ts_labs, ts_vitals, ts_respiratory):
    # Time relative to first ICU admission
    first_time_acidosis = get_inclusion_time(ts_labs) # TODO: optimize so that doesnt have to recalculate
    sofa_scores = sofa_helper.calc_sofa(patient_info, ts_vitals, meds, ts_labs, ts_respiratory).rename({"id" : "Global ICU Stay ID"})

    outcome_lf = patient_info.join(
        first_time_acidosis, on="Global ICU Stay ID", how="inner"
    ).with_columns(
        # Death time relative to inclusion time :
        # (Pre-ICU Length of Stay + time_to_inclusion) = Inclusion time relative to hospitalisation start
        # Hospital Length of stay - inclusion rel to hospitalisation start = time to death relative to inclusion 
        pl.when(pl.col("Mortality in Hospital")).then(
            pl.col("Hospital Length of Stay (days)") - pl.col("Pre-ICU Length of Stay (days)") - pl.col("First Acidosis Time")/(3600*24)
        ).otherwise(None).alias("death_rel_to_inclusion")
    )

    # Compare delta at given time to inclusion 
    # inclusion is the min(time) in this dataset
    calc_delta = lambda col_name : (pl.col(col_name) - pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID")).alias(f"{col_name}_delta_to_inclusion")
    compare_delta = lambda col_name : ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) | ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) & (pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID") == 3))).alias(f"{col_name}_sig_increase")

    sofas = ["sofa_coag", "sofa_liver", "sofa_renal", "sofa_cardio", "sofa_resp"]

    sofa_scores = sofa_scores.join(
        first_time_acidosis,
        on="Global ICU Stay ID",
        how="inner"
    ).filter(
        pl.col("time") >= pl.col("First Acidosis Time")
    ).with_columns(
        [calc_delta(sofa) for sofa in sofas]
    ).with_columns(
        [compare_delta(sofa) for sofa in sofas]
    ).fill_null(False).filter(
        pl.col("sofa_coag_sig_increase") |
        pl.col("sofa_liver_sig_increase") |
        pl.col("sofa_renal_sig_increase") |
        pl.col("sofa_cardio_sig_increase") |
        pl.col("sofa_resp_sig_increase")
    ).filter(
        pl.col("time") == pl.col("time").min().over("Global ICU Stay ID")
    ).rename({
        "time" : "sofa_increase_rel_to_inclusion"
    })

    outcome_lf = outcome_lf.join(sofa_scores, on="Global ICU Stay ID", how="inner")
    #print(sofa_scores.select(pl.col("sofa_coag"), pl.col("sofa_coag_delta_to_inclusion"), pl.col("sofa_coag_sig_increase")).collect())
    
    composition_outcome = outcome_lf.filter(
        (pl.col("sofa_increase_rel_to_inclusion") < 28 * 3600 * 24) &
        (pl.col("death_rel_to_inclusion") < 28)
    )
    print(f"{composition_outcome.select("Global ICU Stay ID").unique().collect().to_series().len()} patients do fit the outcome criteria. ")

    return composition_outcome.select("Global ICU Stay ID").unique()

def select_patients(selection_lf:pl.Series, patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output):
    return tuple([
        patient_information.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        medications.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        diagnoses.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        procedures.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        microbiology.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        ts_labs.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        ts_vitals.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        ts_respiratory.filter(pl.col("Global ICU Stay ID").is_in(selection_lf)),
        ts_intake_output.filter(pl.col("Global ICU Stay ID").is_in(selection_lf))])


## ------------- ANALASIS FUNCTIONS -------------
def encode_hot_ones(lf, cols):
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
        "p-value" : wls.pvalues[1],
        "ci_low": wls.conf_int(alpha=0.05)[1][0],
        "ci_high": wls.conf_int(alpha=0.05)[1][1]
    }

if __name__ == "__main__":
    patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output = load_data()

    print(f"{patient_information.select("Global ICU Stay ID").unique().collect().to_series().len()} unique patients loaded.")

    exposure, control = get_bicarb_exposure_control_24h_after_inclusion(medications, ts_labs)

    print(f"{exposure.select("Global ICU Stay ID").unique().collect().to_series().len()} patients in exposure groupe.")
    print(f"{control.select("Global ICU Stay ID").unique().collect().to_series().len()} patients in casual contrast groupe.")


    #exposure_patient_information, exposure_medications, exposure_diagnoses, exposure_procedures, exposure_microbiology, exposure_ts_labs, exposure_ts_vitals, exposure_ts_respiratory, exposure_ts_intake_output = select_patients(exposure.select("Global ICU Stay ID").collect().to_series(), patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output)
    #control_patient_information, control_medications, control_diagnoses, control_procedures, control_microbiology, control_ts_labs, control_ts_vitals, control_ts_respiratory, control_ts_intake_output = select_patients(control.select("Global ICU Stay ID").collect().to_series(), patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output)
    
    #print("Exposure groupe : ")
    #exposure_outcome_patients = outcome(exposure_patient_information, exposure_medications, exposure_ts_labs, exposure_ts_vitals, exposure_ts_respiratory)
    #print("Control groupe : ")
    #control_outcome_patients = outcome(control_patient_information, control_medications, control_ts_labs, control_ts_vitals, control_ts_respiratory)

    outcome_patients = outcome(patient_information, medications, ts_labs, ts_vitals, ts_respiratory)

    confounders_columns = ["Admission Age (years)", "Admission Weight (kg)"]

    outcome_lf = patient_information.select(
            [pl.col("Global ICU Stay ID")] + 
            [pl.col(column) for column in confounders_columns]
        ).with_columns([
        (
            pl.when(pl.col("Global ICU Stay ID").is_in(exposure.collect().to_series()))
            .then(1)
            .when(pl.col("Global ICU Stay ID").is_in(control.collect().to_series()))
            .then(0)
            .otherwise(None)
        ).alias("Exposure"),
        (
            pl.when(pl.col("Global ICU Stay ID").is_in(outcome_patients.collect().to_series()))
            .then(1)
            .otherwise(0)
        ).alias("Outcome")
    ])
    # Test quality of groupe
    if outcome_lf.filter(pl.col("Exposure").is_null()).select("Global ICU Stay ID").collect().to_series().len() > 0: 
        raise ValueError("There are patients not in exposure or control groupe in the dataset !")
    #print(outcome_lf.collect()) 

    outcome_lf = outcome_lf.filter(pl.col("Admission Weight (kg)").is_not_null())
        

    # Get hot ones on categorial columns
    outcome_df = encode_hot_ones(outcome_lf, confounders_columns)

    #print(outcome_df)

    print("Performing analysis...")
    outcome_np = outcome_df.select("Outcome").to_numpy().ravel()
    exposure_np = outcome_df.select("Exposure").to_numpy().ravel()
    
    confounders_np = outcome_df.select([pl.col(column) for column in outcome_df.columns if column not in ["Global ICU Stay ID", "Outcome", "Exposure"]]).to_numpy()

    #print(confounders_np)

    #print(unajusted_regression(exposure_np, outcome_np))
    x = propensity_score(exposure_np, outcome_np, confounders_np)
    print(x)