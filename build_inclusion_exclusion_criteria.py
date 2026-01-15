
import reprodICU 
import polars as pl

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


#### -------- INCLUSION CRITERIA -------- ##

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
# Patient withz acidemia in ANY LAB within 48h
acidemia_any_lab = ts_labs.filter(
    (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
    (pl.col("pH").struct.field("value") <= 7.2)
).filter(
    (pl.col("Carbon dioxide").struct.field("value") <= 45).any().over("Global ICU Stay ID") &
    (pl.col("Bicarbonate").struct.field("value") <= 20).any().over("Global ICU Stay ID")
)

# Incliusion time : first acidemia lab time
inclusion_time = acidemia_any_lab.group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("inclusion_time_seconds"))

sofa = sofa_helper.calc_sofa(patient_information, ts_vitals, medications, ts_labs, ts_respiratory)
sofa = sofa.rename({"id" : "Global ICU Stay ID"})
# SOFA at any time 
sofa_at_any_time = sofa.filter(pl.col("sofa") >= 4)
# SOFA upto inclusion time
sofa_upto_inclusion_time = sofa.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(pl.col("sofa") >= 4).filter(
    pl.col("time") <= pl.col("inclusion_time_seconds")
    )
# SOFA 48h to inclusion event
sofa_48h_to_inclusion = sofa.join(inclusion_time, on="Global ICU Stay ID", how="inner").filter(
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

####### EXCLUSION CRITERIA ########

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

# Table with columns defining inclusion creteria
iterable_names_dict = {
    "include_adults" : adult_patients.select("Global ICU Stay ID").unique(),
    "include_severe_acidemia_in_48h_one_lab" : acidemia_one_lab.select("Global ICU Stay ID").unique(),
    "include_severe_acidemia_in_48h_any_lab" : acidemia_any_lab.select("Global ICU Stay ID").unique(),
    "include_sofa_at_any_time" : sofa_at_any_time.select("Global ICU Stay ID").unique(),
    "include_sofa_upto_inclusion" : sofa_upto_inclusion_time.select("Global ICU Stay ID").unique(),
    "include_sofa_48h_to_inclusion" : sofa_48h_to_inclusion.select("Global ICU Stay ID").unique(),
    "include_lactate_at_any_time" : lactate_any_time.select("Global ICU Stay ID").unique(),
    "include_lactate_upto_inclusion" : lactate_upto_inclusion.select("Global ICU Stay ID").unique(),
    "include_lactate_48h_to_inclusion" : lactate_48h_to_inclusion.select("Global ICU Stay ID").unique(),
    "exclude_respiratory_acidosis" : respiratory_acidosis.select("Global ICU Stay ID").unique(),
    "exclude_ketoacidosis" : ketoacidosis.select("Global ICU Stay ID").unique(),
    "exclude_prior_RRT_at_any_time" : rrt_at_any_time.select("Global ICU Stay ID").unique(),
    "exclude_prior_RRT_upto_inclusion" : rrt_upto_inclusion.select("Global ICU Stay ID").unique(),
    "exclude_CKD" : ckd.select("Global ICU Stay ID").unique(),
    "exclude_gfr_below_30" : patients_dfg_before_inclusion.select("Global ICU Stay ID").unique(),
}

CRITERIA_TABLE = patient_information.select("Global ICU Stay ID")
for name, df in iterable_names_dict.items():
    print(f"Processing criterion: {name}")
    CRITERIA_TABLE = CRITERIA_TABLE.with_columns([
        pl.when(pl.col("Global ICU Stay ID").is_in(df.collect().to_series()))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias(name)
    ])


print(CRITERIA_TABLE.head().collect())

CRITERIA_TABLE.sink_parquet("inclusion_exclusion_criteria_table.parquet")

