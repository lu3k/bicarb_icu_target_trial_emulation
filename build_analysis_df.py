import polars as pl
from pathlib import Path
import sofa_helper


## -------- helper functions ----------##
# Inclusion time is first severe acidosis time 
def get_inclusion_time(ts_labs):
    first_acidosis_time = ts_labs.filter(
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2) 
    ).group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("first_acidosis_time"))
    
    return first_acidosis_time

# Exposure is bicarb medication 
def get_bicarb_exposure(medications):
    bicarbs = medications.filter(
        pl.col("Drug Name").str.to_lowercase().str.contains("bicarb") |
        pl.col("Drug Name").str.to_lowercase().str.contains("hco") |
        pl.col("Drug Ingredient").str.contains("sodium bicarbonate")
    )
    return bicarbs

## -------- build data -------------- ##

# Load Data
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

# Add inclusion time // we can use inner join, beacause all included patients should have a time defined
patient_information = patient_information.join(
    get_inclusion_time(ts_labs),
    on="Global ICU Stay ID",
    how="inner"
)
# Add bicarb exposure 
_bicarb_exposure = get_bicarb_exposure(medications)
patient_information = patient_information.with_columns([
    pl.when(pl.col("Global ICU Stay ID").is_in(_bicarb_exposure.select("Global ICU Stay ID").unique().collect().to_series()))
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("bicarb_exposure")
])
# Bicarb exposure within 24h of inclusion ? 
_bicarb_exposure = _bicarb_exposure.join(
    patient_information.select("Global ICU Stay ID", "first_acidosis_time"), on="Global ICU Stay ID"
).with_columns(
    # Calculate the Drug start time relative to first acidosis
    (pl.col("Drug Start Relative to Admission (seconds)") - pl.col("first_acidosis_time")).alias("drug_start_rel_to_inclusion"),
).with_columns([
    # Check if is < 24h
    pl.when(pl.col("drug_start_rel_to_inclusion") <= 24*3600).then(1).otherwise(0).alias("exposed_in_24h"),
])
# Add info to patient // use left since some are not defined
patient_information = patient_information.join(
    _bicarb_exposure.select("Global ICU Stay ID", "drug_start_rel_to_inclusion", "exposed_in_24h"), on="Global ICU Stay ID", how="left"
).with_columns(
    # If null patient was not at all exposed, so set to 0 
    pl.col("exposed_in_24h").fill_null(0)
)

# Add FOLLOWUP events :
# Assessing organ failure at day 7 and mortality at day 28 or discharge (whichever comes sooner)

# TODO : Add organ failure
patient_information = patient_information.with_columns(
    # Death time or discharge time relative to inclusion time (since death is also the end of hosptialisation):
    # (Pre-ICU Length of Stay + time_to_inclusion) = Inclusion time relative to hospitalisation start
    # Hospital Length of stay - inclusion rel to hospitalisation start = time to death relative to inclusion 
    (pl.when(pl.col("Mortality in Hospital")).then(pl.lit("death")).otherwise(pl.lit("discharge"))).alias("follow_up_event"),
    (pl.col("Hospital Length of Stay (days)") - pl.col("Pre-ICU Length of Stay (days)") - pl.col("first_acidosis_time")/(3600*24)).alias("time_to_follow_up_event_rel_to_inclusion")
)

# Add OUTCOME events
_sofa_scores = sofa_helper.calc_sofa(patient_information, ts_vitals, medications, ts_labs, ts_respiratory).rename({"id" : "Global ICU Stay ID"})
# Compare delta at given time to inclusion 
# inclusion is the min(time) in this dataset
calc_delta = lambda col_name : (pl.col(col_name) - pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID")).alias(f"{col_name}_delta_to_inclusion")
compare_delta = lambda col_name : ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) | ((pl.col(f"{col_name}_delta_to_inclusion") >= 2) & (pl.col(col_name).sort_by("time").drop_nulls().first().over("Global ICU Stay ID") == 3))).alias(f"{col_name}_sig_increase")
_sofa_scores = _sofa_scores.join(
    patient_information.select("Global ICU Stay ID", "first_acidosis_time"),
    on="Global ICU Stay ID",
    how="inner"
).filter(
    pl.col("time") >= pl.col("first_acidosis_time")
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

# Join sofa time to patient information / Using left since not all patients have an increase in sofa score
patient_information = patient_information.join(
    _sofa_scores.select("Global ICU Stay ID", "sofa_increase_rel_to_inclusion"),
    on="Global ICU Stay ID",
    how="left"
).with_columns(
    (pl.col("sofa_increase_rel_to_inclusion") <= 28 * 3600 * 24).alias("sofa_increase_in_28d").fill_null(False),
    ((pl.col("follow_up_event") == "death") & (pl.col("time_to_follow_up_event_rel_to_inclusion") <= 28)).alias("death_in_28d")
).with_columns(
    (pl.col("sofa_increase_in_28d") & pl.col("death_in_28d")).alias("cumulative_outcome")
).with_columns(
    pl.col("sofa_increase_in_28d").cast(pl.Int8),
    pl.col("death_in_28d").cast(pl.Int8),
    pl.col("cumulative_outcome").cast(pl.Int8),
)

# We now have the patient_information lf with following columns used for analysis: 
# inclusion_time, 
# exposed_in_24h (0, 1)
# follow_up_event (death / discharge)
# time_to_follow_up_event_rel_to_inclusion
# sofa_increase_in_28d, death_in_28d, cumulative_outcome (0, 1)


## ------- interaction functions ------ ##
def get_exposure_np():
    return patient_information.select("exposed_in_24h").collect().to_numpy().ravel()
def get_outcome_np():
    return patient_information.select("cumulative_outcome").collect().to_numpy().ravel()
def get_confounders_np(confounder_columns):
    return patient_information.select(confounder_columns).collect().to_numpy()
def get_kaplan_meier_pd(censor_time=28):
    # censor patients if no event after a certain time to censor time
    return patient_information.select(
        "exposed_in_24h",
        "cumulative_outcome",
        "time_to_follow_up_event_rel_to_inclusion"
    ).with_columns(
        pl.when(pl.col("time_to_follow_up_event_rel_to_inclusion") > censor_time)
        .then(pl.lit(censor_time))
        .otherwise(pl.col("time_to_follow_up_event_rel_to_inclusion"))
        .alias("time_to_follow_up_event_rel_to_inclusion")
    ).collect().to_pandas()