# This script takes the reprodICU data and builds an analysis table to analyise the effect of bicarbonate administration on outcome in acidemic patients.

import polars as pl
import reprodICU
import sofa_helper
import edfg_helper

patient_information = reprodICU.patient_information
medications = reprodICU.medications
diagnoses = reprodICU.diagnoses
procedures = reprodICU.procedures
microbiology = reprodICU.microbiology
ts_labs = reprodICU.timeseries_labs
ts_vitals = reprodICU.timeseries_vitals
ts_respiratory = reprodICU.timeseries_respiratory
ts_intake_output = reprodICU.timeseries_intakeoutput

# Clean Global ICU Stay ID columns
ts_labs = ts_labs.with_columns(
    pl.col("Global ICU Stay ID").str.replace(r"\.0$", "").alias("Global ICU Stay ID")
) 

ts_sofa = sofa_helper.get_sofa(patient_information, ts_vitals, medications, ts_labs, ts_respiratory).rename({"id" : "Global ICU Stay ID"})

def generate_criteria_table(iterable_names_dict: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """
    Generate a criteria table with boolean columns for each criterion in the input dictionary.
    Each column indicates whether the patient meets the criterion.
    """
    CRITERIA_TABLE = patient_information.select("Global ICU Stay ID")
    for name, df in iterable_names_dict.items():
        print(f"Processing criterion: {name}")
        df = df.select("Global ICU Stay ID").unique()
        CRITERIA_TABLE = CRITERIA_TABLE.with_columns([
            pl.when(pl.col("Global ICU Stay ID").is_in(df.collect().to_series()))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias(name)
        ])
    return CRITERIA_TABLE

def get_inclusion_table() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Build the inclusion criteria table and return it along with the inclusion time table.
    Inclusion criteria include adult status, severe acidemia, SOFA score, and lactate levels.
    """

    # Adult patients
    adult_patients = patient_information.filter(
        pl.col("Admission Age (years)") >= 18
    )

    # Patients with acidemia (pH <= 7.2, CO2 <= 45, Bicarb <= 20) within 48h of admission ONE LAB
    acidemia_one_lab = ts_labs.filter(
        pl.col("pH").is_not_null() & pl.col("Carbon dioxide").is_not_null() & pl.col("Bicarbonate").is_not_null() &
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2) &
        (pl.col("Carbon dioxide").struct.field("value") <= 45) &
        (pl.col("Bicarbonate").struct.field("value") <= 20)
    )               

    # Inclusion time : first acidemia lab time
    inclusion_time = acidemia_one_lab.group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("inclusion_time_seconds"))

    # SOFA at any time 
    sofa_at_any_time = ts_sofa.filter(pl.col("sofa") >= 4)
    # SOFA upto inclusion time
    sofa_upto_inclusion_time = ts_sofa.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(pl.col("sofa") >= 4).filter(
        pl.col("time") <= pl.col("inclusion_time_seconds")
        )
    # SOFA 48h to inclusion event
    sofa_48h_to_inclusion = ts_sofa.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(
        (pl.col("sofa") >= 4) &
        (pl.col("time") >= (pl.col("inclusion_time_seconds") - 48 * 3600)) &
        (pl.col("time") <= pl.col("inclusion_time_seconds"))
    )

    # Lactate over 2 at any time
    lactate_any_time = ts_labs.filter(pl.col("Lactate").struct.field("value") >= 2)
    # Lactate upto inclusion time 
    lactate_upto_inclusion = ts_labs.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(
        (pl.col("Lactate").struct.field("value") >= 2) &
        (pl.col("Time Relative to Admission (seconds)") <= pl.col("inclusion_time_seconds"))
    )
    # Lactate 48h to inclusion time
    lactate_48h_to_inclusion = ts_labs.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(
        (pl.col("Lactate").struct.field("value") >= 2) &
        (pl.col("Time Relative to Admission (seconds)") >= (pl.col("inclusion_time_seconds") - 48 * 3600)) &
        (pl.col("Time Relative to Admission (seconds)") <= pl.col("inclusion_time_seconds"))
    )

    return generate_criteria_table({
        "include_adults" : adult_patients,
        "include_severe_acidemia_in_48h" : acidemia_one_lab,
        "include_sofa_at_any_time" : sofa_at_any_time,
        "include_sofa_upto_inclusion" : sofa_upto_inclusion_time,
        "include_sofa_48h_to_inclusion" : sofa_48h_to_inclusion,
        "include_lactate_at_any_time" : lactate_any_time,
        "include_lactate_upto_inclusion" : lactate_upto_inclusion,
        "include_lactate_48h_to_inclusion" : lactate_48h_to_inclusion,   
    }), inclusion_time

def get_exclusion_table(inclusion_time: pl.LazyFrame) -> pl.LazyFrame:
    """
    Build the exclusion criteria table based on respiratory acidosis, ketoacidosis,
    prior RRT, CKD, and GFR below 30 before inclusion.
    """

    # Respiratory acidosis (CO2 >= 45)
    respiratory_acidosis = ts_labs.filter(pl.col("Carbon dioxide").struct.field("value") >= 45)

    # Kétoacidosis (Ketones >= 3)
    ketoacidosis = ts_labs.filter(pl.col("Ketones").struct.field("value") >= 3)

    ## Ignore volume loss for now

    # Prior RRT
    ## ICD10 codes for RRT
    #5A1D00Z — Performance of urinary filtration, single (standard intermittent hemodialysis)
    #5A1D60Z — Performance of urinary filtration, intermittent
    #5A1D70Z — Performance of urinary filtration, continuous
    #5A1D80Z — Performance of urinary filtration, continuous with replacement fluid
    #5A1D80Z — because ICD-PCS groups these under continuous filtration with replacement fluid.
    #5A1D90Z — Performance of urinary filtration for fluid removal only
    #5A1D90Z — Drainage of peritoneal cavity using percutaneous approach (PD fluid removal)
    #3E1M39Z — Introduction of dialysis solution into peritoneal cavity via percutaneous approach (PD initiation)
    icd10_rrt = ["5A1D00Z", "5A1D60Z", "5A1D70Z", "5A1D80Z", "5A1D90Z", "3E1M39Z"]

    ## ICD9 codes for RRT
    #39.95 — Hemodialysis (used for CRRT as well)
    #54.98 — Other peritoneal dialysis or ultrafiltration procedure
    #54.98 — Other peritoneal dialysis
    icd9_rrt = ["39.95", "54.98"]

    rrt_lf = procedures.filter(
        pl.col("Procedure ICD Code").is_not_null() &
        (
            ((pl.col("Procedure ICD Code Version") == 10) & pl.col("Procedure ICD Code").is_in(icd10_rrt)) |
            ((pl.col("Procedure ICD Code Version") == 9) & pl.col("Procedure ICD Code").is_in(icd9_rrt)) 
        )
    ).join(patient_information.select("Global Person ID", "Global ICU Stay ID"), on="Global Person ID", how="inner").drop("Global ICU Stay ID").rename({"Global ICU Stay ID_right" : "Global ICU Stay ID"})

    rrt_at_any_time = rrt_lf.select("Global ICU Stay ID").unique()
    rrt_upto_inclusion = rrt_lf.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(
        pl.col("Procedure Start Relative to Admission (seconds)") <= pl.col("inclusion_time_seconds")
    )

    # Diagnosed CKD Stage 4 (N18.4, N18.5), AKI (N17.-) or GFR <30 at time of acidosis
    # CKD Stage 4 (N18.4, N18.5), AKI (N17.-)
    icd10_codes = ["N18.4", "N18.5", "N17", "N17.0", "N17.1", "N17.2", "N17.8", "N17.9"]
    icd9_codes = ["585.4", "585.5", "584", "584.5", "584.6", "584.7", "584.8", "584.9"]

    ckd = diagnoses.filter(
        ((pl.col("Diagnosis ICD Code Version (source)") == "ICD-9") & pl.col("Diagnosis ICD-9 Code").is_in(icd9_codes)) |
        ((pl.col("Diagnosis ICD Code Version (source)") == "ICD-10") & pl.col("Diagnosis ICD-10 Code").is_in(icd10_codes))
    ).join(
        inclusion_time, on="Global ICU Stay ID", how="inner"
    ).filter(
        # Either before inclusion time
        (pl.col("Diagnosis Start Relative to Admission (seconds)") <= pl.col("inclusion_time_seconds"))
        # OR assuming that if diagnosis time is missing, it was present at admission
        | pl.col("Diagnosis Start Relative to Admission (seconds)").is_null()
    )

    edfg = edfg_helper.eDFG_ckd_epi(patient_information, ts_labs)
    patients_dfg_before_inclusion = edfg.filter(pl.col("eDFG CKD-EPI") <= 30 ).join(
        inclusion_time,
        on="Global ICU Stay ID",
        how="inner"
    ).filter(
        pl.col("Time Relative to Admission (seconds)") <= pl.col("inclusion_time_seconds")
    )

    return generate_criteria_table({
        "exclude_respiratory_acidosis" : respiratory_acidosis,
        "exclude_ketoacidosis" : ketoacidosis,
        "exclude_prior_RRT_at_any_time" : rrt_at_any_time,
        "exclude_prior_RRT_upto_inclusion" : rrt_upto_inclusion,
        "exclude_CKD" : ckd,
        "exclude_gfr_below_30" : patients_dfg_before_inclusion,
    })

def get_follow_up_outcome_table(inclusion_time: pl.LazyFrame) -> pl.LazyFrame:
    """
    Build the follow-up and outcome table, including death and SOFA score increase within 28 days.
    """

    print("Processing follow-up and outcome data...")

    # TODO : Add organ failure
    follow_up = patient_information.join(inclusion_time, on="Global ICU Stay ID", how="inner")
    follow_up = follow_up.with_columns(
        # Death time or discharge time relative to inclusion time (since death is also the end of hosptialisation):
        # (Pre-ICU Length of Stay + time_to_inclusion) = Inclusion time relative to hospitalisation start
        # Hospital Length of stay - inclusion rel to hospitalisation start = time to death relative to inclusion 
        (pl.when(pl.col("Mortality in Hospital")).then(pl.lit("death")).otherwise(pl.lit("discharge"))).alias("follow_up_event"),
        (pl.col("Hospital Length of Stay (days)") - pl.col("Pre-ICU Length of Stay (days)") - pl.col("inclusion_time_seconds")/(3600*24)).alias("time_to_follow_up_event_rel_to_inclusion")
    ).select("Global ICU Stay ID", "follow_up_event", "time_to_follow_up_event_rel_to_inclusion")

    ### OUTCOME CRITERIA ###
    calc_delta = lambda col_name : (pl.col(col_name) - pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID")).alias(f"{col_name}_delta_to_inclusion")
    compare_delta = lambda col_name : ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) | ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) & (pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID") == 3))).alias(f"{col_name}_sig_increase")
    sofa_sig_increase = ts_sofa.join(
        inclusion_time,
        on="Global ICU Stay ID",
        how="inner"
    ).filter(
        pl.col("time") >= pl.col("inclusion_time_seconds")
    ).with_columns(
        [calc_delta(sofa) for sofa in ["sofa_coag", "sofa_liver", "sofa_renal", "sofa_cardio", "sofa_resp"]]
    ).with_columns(
        [compare_delta(sofa) for sofa in ["sofa_coag", "sofa_liver", "sofa_renal", "sofa_cardio", "sofa_resp"]]
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

    OUTCOME_TABLE = patient_information.select("Global ICU Stay ID").unique().join(
        follow_up, on="Global ICU Stay ID", how="left"
    ).join(
        sofa_sig_increase.select("Global ICU Stay ID", "sofa_increase_rel_to_inclusion"),
        on="Global ICU Stay ID",
        how="left"
    ).with_columns(
        (pl.col("sofa_increase_rel_to_inclusion") <= 28 * 3600 * 24).alias("sofa_increase_in_28d").fill_null(False),
        ((pl.col("follow_up_event") == "death") & (pl.col("time_to_follow_up_event_rel_to_inclusion") <= 28)).alias("death_in_28d")
    ).with_columns(
        (pl.col("sofa_increase_in_28d") & pl.col("death_in_28d")).alias("death_and_sofa_increase_28d")
    ).with_columns(
        pl.col("sofa_increase_in_28d").cast(pl.Int8),
        pl.col("death_in_28d").cast(pl.Int8),
        pl.col("death_and_sofa_increase_28d").cast(pl.Int8),
    )

    return OUTCOME_TABLE

def get_exposure_table(inclusion_time: pl.LazyFrame) -> pl.LazyFrame:
    """
    Build the exposure table indicating bicarbonate administration and timing relative to inclusion.
    """

    print("Processing bicarbonate exposure data...")

    bicarbonate_medications = medications.filter(
        pl.col("Drug Name").str.to_lowercase().str.contains("bicarb") |
        pl.col("Drug Name").str.to_lowercase().str.contains("hco") |
        pl.col("Drug Ingredient").str.contains("sodium bicarbonate")
    )
    first_bicarbonate_administration = bicarbonate_medications.filter(
        # Only get first bicarb administration
        pl.col("Drug Start Relative to Admission (seconds)") == pl.col("Drug Start Relative to Admission (seconds)").min().over("Global ICU Stay ID")
    )

    EXPOSURE_TABLE = patient_information.select("Global ICU Stay ID").unique().with_columns([
        pl.when(pl.col("Global ICU Stay ID").is_in(bicarbonate_medications.select("Global ICU Stay ID").unique().collect().to_series()))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("bicarbonate_exposure")
    ]).join(
        first_bicarbonate_administration.select(
            "Global ICU Stay ID", "Drug Start Relative to Admission (seconds)"
        ).rename(
            {"Drug Start Relative to Admission (seconds)" : "first_bicarbonate_administration_seconds"}
        ), on="Global ICU Stay ID", how="left"
    ).join(
        inclusion_time, on="Global ICU Stay ID", how="left"
    ).with_columns([
        pl.when(pl.col("bicarbonate_exposure") == True)
        .then(pl.col("first_bicarbonate_administration_seconds") - pl.col("inclusion_time_seconds"))
        .otherwise(pl.lit(None))
        .alias("time_to_bicarbonate_administration_seconds")
    ]).with_columns(
        pl.when(
            (pl.col("bicarbonate_exposure") == True) & 
            (pl.col("time_to_bicarbonate_administration_seconds") <= 24*3600)
        ).then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("exposed_in_24h")
    )
    
    return EXPOSURE_TABLE.select(
        "Global ICU Stay ID", 
        "bicarbonate_exposure", 
        "first_bicarbonate_administration_seconds", 
        "time_to_bicarbonate_administration_seconds", 
        "exposed_in_24h"
    )

def get_analysis_table() -> pl.LazyFrame:
    """
    Build and return the final analysis table by joining inclusion, exclusion, outcome, and exposure tables.
    """

    INCLUSION_TABLE, inclusion_time = get_inclusion_table()
    EXCLUSION_TABLE = get_exclusion_table(inclusion_time)
    FOLLOW_UP_TABLE = get_follow_up_outcome_table(inclusion_time)
    EXPOSURE_TABLE = get_exposure_table(inclusion_time)

    ANALYSIS_TABLE = INCLUSION_TABLE.join(
        EXCLUSION_TABLE, on="Global ICU Stay ID", how="left" 
    ).join(
        FOLLOW_UP_TABLE, on="Global ICU Stay ID", how="left"
    ).join(
        EXPOSURE_TABLE, on="Global ICU Stay ID", how="left"
    )

    return ANALYSIS_TABLE

ANALYSIS_TABLE = get_analysis_table()
ANALYSIS_TABLE.sink_parquet("bicarbicu_analysis_table.parquet")
