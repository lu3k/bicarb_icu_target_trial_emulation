# Creating a cohort for the bicarb emulation study
# Format is a pipeline. Each function takes in all the LazyFrames of reprodICU and includes / excludes patients. 
# then passing on to the next. 
# 

import polars as pl
import reprodICU
from functools import wraps
from pathlib import Path
import tempfile
import os

import sofa_helper
import edfg_helper

# Pipeline element that should be used to pipe the cohort
# IMPLEMENT MAGIC CONCEPTS ONCE HiRED AVAILABLE
def my_cohort_pipe_element(filter_column="Global ICU Stay ID", how="semi", debug=None):
    def decorator(function):
        @wraps(function)
        def wrapper(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) -> tuple[pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame]:
            
            if debug: print(f"[{debug}] Receving {patient_information.select("Global ICU Stay ID").unique().collect().to_series().len()} unique patients.")

            # Apply inclusion criteria
            filtered_lf = function(patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output)
            filtered_ids = filtered_lf.select(filter_column).unique()
            
            if debug: 
                if how=="semi": print(f"[{debug}] Saving {filtered_ids.collect().to_series().len()} unique patients.")
                if how=="anti": print(f"[{debug}] Excluding {filtered_ids.collect().to_series().len()} unique patients.")

            # Apply to other LazyFrames
            patient_information =  patient_information.join(filtered_ids, on=filter_column, how=how)
            medications = medications.join(filtered_ids, on=filter_column, how=how)
            diagnoses =  diagnoses.join(filtered_ids, on=filter_column, how=how)
            procedures =  procedures.join(filtered_ids, on=filter_column, how=how)
            microbiology =  microbiology.join(filtered_ids, on=filter_column, how=how)
            ts_labs =  ts_labs.join(filtered_ids, on=filter_column, how=how)
            ts_vitals =  ts_vitals.join(filtered_ids, on=filter_column, how=how)
            ts_respiratory =  ts_respiratory.join(filtered_ids, on=filter_column, how=how)
            ts_intake_output =  ts_intake_output.join(filtered_ids, on=filter_column, how=how)
            
            return patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output
        return wrapper
    return decorator
## --- Inclusion functions --- ##
# Only adults
@my_cohort_pipe_element(debug="Adults")
def include_adults(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame):
    # Apply inclusion criteria
    return patient_information.filter(
        pl.col("Admission Age (years)") >= 18
    )
    
# Severe acidemia within 48hrs of ICU admission
# (pH≤7.20, paCO2≤45mmHg, HCO3 ≤ 20 mmol/L)
# THis function is replaced by the one below checking for acidosis with CO2 and CHO3 at any time, since was to restrictive
@my_cohort_pipe_element(debug="Severe acidemia in 48h of admission in one lab")
def include_severe_acidemia_in_48h_one_lab(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply inclusion criteria 
    return ts_labs.filter(
        pl.col("pH").is_not_null() & pl.col("Carbon dioxide").is_not_null() & pl.col("Bicarbonate").is_not_null() &
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2) &
        (pl.col("Carbon dioxide").struct.field("value") <= 45) &
        (pl.col("Bicarbonate").struct.field("value") <= 20)
    )

@my_cohort_pipe_element(debug="Severe acidemia in 48h of admission in any lab")
def include_severe_acidemia_in_48h(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply inclusion criteria 
    return ts_labs.filter(
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2)
    ).filter(
        (pl.col("Carbon dioxide").struct.field("value") <= 45).any().over("Global ICU Stay ID") &
        (pl.col("Bicarbonate").struct.field("value") <= 20).any().over("Global ICU Stay ID")
    )
    
# SOFA score ≥ 4 (worst within 48hrs upto inclusion event)
# TODO: UPTO INCLUSION EVENT
@my_cohort_pipe_element(debug="Sofa over 4")
def include_sofa_over_4(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply inclusion criteria

    ## currently not possible due to MAGICCONCEPTS issues (waiting for HiRED)
    #sofa = reprodICU.utils.scores.SOFA(patient_information, ts_vitals, ts_labs, ts_respiratory, ts_intake_output, medications, ts_respiratory)
    
    ## MEANWHILE : 
    sofa = sofa_helper.calc_sofa(patient_information, ts_vitals, medications, ts_labs, ts_respiratory)
    return sofa.rename({"id" : "Global ICU Stay ID"}).filter(pl.col("sofa") >= 4)

# Lactate ≥ 2 mmol/L (worst within 48hrs up to inclusion event)
# TODO: WITHIN 48h of inclusion event
@my_cohort_pipe_element(debug="Lactate > 2")
def include_lactate_over_2(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply inclusion criteria
    return ts_labs.filter(pl.col("Lactate").struct.field("value") >= 2)



## -- EXCLUDE -- ##

# Respiratory acidosis (paCO2≥45mmHg)
@my_cohort_pipe_element(how="anti", debug="Respiratory acidosis")
def exclude_respiratory_acidosis(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply exclusion criteria
    return ts_labs.filter(pl.col("Carbon dioxide").struct.field("value") >= 45)

# Ketoacidosis (serum ketones > TBD)
# for now ketones > 1000 (include all..)
@my_cohort_pipe_element(how="anti", debug="Ketoacidosis")
def exclude_ketoacidosis(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply exclusion criteria
    return ts_labs.filter(pl.col("Ketones").struct.field("value") > 1000)

# Documented volume loss ≥ 1500 mL/d
@my_cohort_pipe_element(how="anti", debug="Volumne loss over 1500ml")
def exclude_volume_loss_over_1500(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply exclusion criteria
    # TODO : check if premade function in reprodICU !
    #urine_out = reprodICU.utils.clinical.URINE_OUTPUT(patient_information, ts_intake_output)
    #print(urine_out.head().collect())

    volume_loss_lf = (
        ts_intake_output.
        with_columns([
            (pl.sum_horizontal(["Fluid intake", "Fluid intake enteral tube", "Fluid intake intravascular", "Fluid intake nasogastric tube", "Fluid intake oral"])).alias("total_fluid_intake"),
            (pl.col("Time Relative to Admission (seconds)") // (24 * 3600)).alias("time_in_days")
        ]).group_by(["Global ICU Stay ID", "time_in_days"])
        .agg([
            pl.col("total_fluid_intake").sum().alias("fluid_intake_sum"),
            pl.col("Urine output").sum().alias("urine_output_sum"),
        ]).sort(["Global ICU Stay ID", "time_in_days"])
        .with_columns([
            (pl.col("fluid_intake_sum") - pl.col("urine_output_sum")).alias("volume_change_day")
        ])
    )
    return volume_loss_lf.filter(pl.col("volume_change_day") <= -1500)
    
# Any RRT prior to acidosis
@my_cohort_pipe_element(how="anti", debug="Prior RRT")
def exclude_prior_RRT(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    # Apply exclusion criteria
    
    # TODO : Once access to other DB 
    #x = reprodICU.utils.clinical.RENAL_REPLACEMENT_THERAPY_FREE_DAYS()

    # MEANWHILE : 
    # first time of acidosis : 
    first_acidosis_time = ts_labs.filter(
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2)
    ).group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("First Acidosis Time"))

    # select RRT
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
    )
    ##### PROBLEM : no ICU stay ID.....
    rrt_at_any_time = rrt_lf.select("Global Person ID").unique().collect().to_series()
    print(f"[!] NO ICU STAY ID and treatment time available for RRT, can not exclude previous RRT patients....")
    print(f"[+] {rrt_at_any_time.len()} patients having RRT at any time")

    ### IMPUTING ICU STAY
    print(f"[!] Imputing ICU stay based on global hospitalisation stay ID / !!! acidosis is based on ICU admission not hospital admission time !! Possible error " )
    print(f"[!] No procedure start available.")

    exclude = first_acidosis_time.join(
        patient_information.select(pl.col("Global ICU Stay ID"), pl.col("Global Hospital Stay ID")), 
        on="Global ICU Stay ID",
        how="inner"
    ).join(
        rrt_lf,
        on="Global Hospital Stay ID",
        how="inner"
    )
    print(f"[+] Excluding based on RRT during entire stay, ! not all global person IDs do have an ICU Stay ID. There will be a difference in the numnber of patients.")
    return exclude


# Diagnosed CKD Stage 4 (N18.4, N18.5), AKI (N17.-) or GFR <30 at time of acidosis
@my_cohort_pipe_element(how="anti", debug="Diagnosed CKD Stage 4, AKI, GFR<30 at time of acidosis")
def exclude_CKD_AKI_GFR_sub30(patient_information: pl.LazyFrame, medications: pl.LazyFrame, diagnoses:pl.LazyFrame, procedures:pl.LazyFrame, microbiology: pl.LazyFrame, ts_labs: pl.LazyFrame, ts_vitals:pl.LazyFrame, ts_respiratory:pl.LazyFrame, ts_intake_output:pl.LazyFrame) :
    
    ## time of acidois
    first_acidosis_time = ts_labs.filter(
        (pl.col("Time Relative to Admission (seconds)") <= 48 * 3600) &
        (pl.col("pH").struct.field("value") <= 7.2)
    ).group_by("Global ICU Stay ID").agg(pl.col("Time Relative to Admission (seconds)").min().alias("First Acidosis Time"))
    #print(f"Patients with acidosis before ICU : {first_acidosis_time.filter(pl.col("First Acidosis Time") < 0).select("Global ICU Stay ID").unique().collect().to_series().len()}")

    # Apply exclusion criteria

    ## prior CKD
    # CKD Stage 4 (N18.4, N18.5), AKI (N17.-)
    icd10_codes = ["N18.4", "N18.5", "N17", "N17.0", "N17.1", "N17.2", "N17.8", "N17.9"]
    icd9_codes = ["585.4", "585.5", "584", "584.5", "584.6", "584.7", "584.8", "584.9"]

    ckd = diagnoses.filter(
        ((pl.col("Diagnosis ICD Code Version (source)") == "ICD-9") & pl.col("Diagnosis ICD-9 Code").is_in(icd9_codes)) |
        ((pl.col("Diagnosis ICD Code Version (source)") == "ICD-10") & pl.col("Diagnosis ICD-10 Code").is_in(icd10_codes))
    )
    ## assuming that if no diagnosis time = prior disease / < 0s to admission 
    prior_ckd = ckd.filter(
        pl.col("Diagnosis Start Relative to Admission (seconds)").is_null() | 
        (pl.col("Diagnosis Start Relative to Admission (seconds)") < 0)
    )
    diagnosed_ckd_before_acidosis = ckd.filter(
        pl.col("Diagnosis Start Relative to Admission (seconds)").is_not_null()
    ).join(
        first_acidosis_time, on="Global ICU Stay ID", how="inner"
    ).filter(
        pl.col("Diagnosis Start Relative to Admission (seconds)") <= pl.col("First Acidosis Time")
    )

    print(f"Patients with prior CKD : {prior_ckd.select("Global ICU Stay ID").unique().collect().to_series().len()}")
    print(f"Patients with CKD diagnosis before first acidosis time : {diagnosed_ckd_before_acidosis.select("Global ICU Stay ID").unique().collect().to_series().len()}")


    ## eGFR<30
    # reprodICU not working for me on GFR, using own function 
    edfg = edfg_helper.eDFG_ckd_epi(patient_information, ts_labs)
    
    edfg = edfg.filter(
        pl.col("eDFG CKD-EPI") <= 30 
    ).join(
        first_acidosis_time,
        on="Global ICU Stay ID",
        how="inner"
    ).filter(
        pl.col("Time Relative to Admission (seconds)") <= pl.col("First Acidosis Time")
    )

    print(f"Patients with eGFR<30 before before acidosis time : {edfg.select("Global ICU Stay ID").unique().collect().to_series().len()}")
    

    # join 
    exclude_patients = pl.concat([
        prior_ckd.select("Global ICU Stay ID"),
        diagnosed_ckd_before_acidosis.select("Global ICU Stay ID"),
        edfg.select("Global ICU Stay ID")
    ], how="vertical")
    return exclude_patients


def build_cohort():
    inclusion_functions = [include_adults, include_severe_acidemia_in_48h, include_sofa_over_4, include_lactate_over_2]
    exclusion_functions = [exclude_respiratory_acidosis, exclude_ketoacidosis, exclude_volume_loss_over_1500, exclude_prior_RRT, exclude_CKD_AKI_GFR_sub30]

    patient_information = reprodICU.patient_information
    medications = reprodICU.medications
    diagnoses = reprodICU.diagnoses
    procedures = reprodICU.procedures
    microbiology = reprodICU.microbiology
    ts_labs = reprodICU.timeseries_labs
    ts_vitals = reprodICU.timeseries_vitals
    ts_respiratory = reprodICU.timeseries_respiratory
    ts_intake_output = reprodICU.timeseries_intakeoutput

    temp_dir = Path(tempfile.gettempdir()) / "cohort_pipeline"
    temp_dir.mkdir(exist_ok=True)

    for step, pipe_element in enumerate(inclusion_functions + exclusion_functions):
        patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output = pipe_element(patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output)

        print(f"[Cohort] Sinking step {step} / {pipe_element.__name__}...")
        patient_information.sink_parquet(temp_dir / f"step_{step}_patient_information.parquet")
        medications.sink_parquet(temp_dir / f"step_{step}_medications.parquet")
        diagnoses.sink_parquet(temp_dir / f"step_{step}_diagnoses.parquet")
        procedures.sink_parquet(temp_dir / f"step_{step}_procedures.parquet")
        microbiology.sink_parquet(temp_dir / f"step_{step}_microbiology.parquet")
        ts_labs.sink_parquet(temp_dir / f"step_{step}_ts_labs.parquet")
        ts_vitals.sink_parquet(temp_dir / f"step_{step}_ts_vitals.parquet")
        ts_respiratory.sink_parquet(temp_dir / f"step_{step}_ts_respiratory.parquet")
        ts_intake_output.sink_parquet(temp_dir / f"step_{step}_ts_intake_output.parquet")

        print(f"[Cohort] Reloading step {step} / {pipe_element.__name__}...")
        patient_information = pl.scan_parquet(temp_dir / f"step_{step}_patient_information.parquet")
        medications = pl.scan_parquet(temp_dir / f"step_{step}_medications.parquet")
        diagnoses = pl.scan_parquet(temp_dir / f"step_{step}_diagnoses.parquet")
        procedures = pl.scan_parquet(temp_dir / f"step_{step}_procedures.parquet")
        microbiology = pl.scan_parquet(temp_dir / f"step_{step}_microbiology.parquet")
        ts_labs = pl.scan_parquet(temp_dir / f"step_{step}_ts_labs.parquet")
        ts_vitals = pl.scan_parquet(temp_dir / f"step_{step}_ts_vitals.parquet")
        ts_respiratory = pl.scan_parquet(temp_dir / f"step_{step}_ts_respiratory.parquet")
        ts_intake_output = pl.scan_parquet(temp_dir / f"step_{step}_ts_intake_output.parquet")

    print(f"[Cohort] Final cohort length = {patient_information.select(pl.col('Global ICU Stay ID')).unique().collect().to_series().len()}")

    print("[Cohort] Copying final parquet to output_directory")
    final_path = Path("bicarb_data")
    os.rename(temp_dir / f"step_{step}_patient_information.parquet", final_path / "patient_information.parquet")
    os.rename(temp_dir / f"step_{step}_medications.parquet", final_path / "medications.parquet")
    os.rename(temp_dir / f"step_{step}_diagnoses.parquet", final_path / "diagnoses.parquet")
    os.rename(temp_dir / f"step_{step}_procedures.parquet", final_path / "procedures.parquet")
    os.rename(temp_dir / f"step_{step}_microbiology.parquet", final_path / "microbiology.parquet")
    os.rename(temp_dir / f"step_{step}_ts_labs.parquet", final_path / "ts_labs.parquet")
    os.rename(temp_dir / f"step_{step}_ts_vitals.parquet", final_path / "ts_vitals.parquet")
    os.rename(temp_dir / f"step_{step}_ts_respiratory.parquet", final_path / "ts_respiratory.parquet")
    os.rename(temp_dir / f"step_{step}_ts_intake_output.parquet", final_path / "ts_intake_output.parquet")

def test():
    patient_information = reprodICU.patient_information
    medications = reprodICU.medications
    diagnoses = reprodICU.diagnoses
    procedures = reprodICU.procedures
    microbiology = reprodICU.microbiology
    ts_labs = reprodICU.timeseries_labs
    ts_vitals = reprodICU.timeseries_vitals
    ts_respiratory = reprodICU.timeseries_respiratory
    ts_intake_output = reprodICU.timeseries_intakeoutput

    patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output = include_severe_acidemia_in_48h(patient_information, medications, diagnoses, procedures, microbiology, ts_labs, ts_vitals, ts_respiratory, ts_intake_output)


if __name__ == "__main__":
    build_cohort()