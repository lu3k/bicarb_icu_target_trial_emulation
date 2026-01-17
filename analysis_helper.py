import polars as pl
import reprodICU

import sofa_helper

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
        ), on="Global ICU Stay ID", how="left")

INCLUSION_TABLE = pl.read_parquet("inclusion_exclusion_criteria_table_one_lab.parquet")
ANALYSIS_TABLE = INCLUSION_TABLE.lazy().join(EXPOSURE_TABLE, on="Global ICU Stay ID", how="left")
ANALYSIS_TABLE = ANALYSIS_TABLE.with_columns([
    pl.when(pl.col("bicarbonate_exposure") == True)
    .then(pl.col("first_bicarbonate_administration_seconds") - pl.col("inclusion_time_seconds_one_lab"))
    .otherwise(pl.lit(None))
    .alias("time_to_bicarbonate_administration_seconds")
])
ANALYSIS_TABLE = ANALYSIS_TABLE.with_columns(
    pl.when(
        (pl.col("bicarbonate_exposure") == True) & 
        (pl.col("time_to_bicarbonate_administration_seconds") <= 24*3600)
    ).then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("exposed_in_24h")
)

# Loose inclusion based on labs at any time and acidemia any lab in 48h
loose_inclusion_criteria = ["include_adults", "include_severe_acidemia_in_48h_one_lab", "include_sofa_at_any_time", "include_lactate_at_any_time"]
# Strict inclusion based on labs 48h prior to admission and acidemia in one lab in 48h
# ATTENTION LABS BASED ON INCLUSION TIME BASED ON ANY ACIDEMIA TIME - FIX LATER
strict_inclusion_criteria = ["include_adults", "include_severe_acidemia_in_48h_one_lab", "include_sofa_48h_to_inclusion", "include_lactate_48h_to_inclusion"]
# Exclusion criteria
loose_exclusion_criteria = ["exclude_respiratory_acidosis", "exclude_ketoacidosis", "exclude_prior_RRT_upto_inclusion", "exclude_CKD"]
strict_exclusion_criteria = ["exclude_respiratory_acidosis", "exclude_ketoacidosis", "exclude_prior_RRT_at_any_time", "exclude_CKD"]

def filter_analysis_table(table, inclusion_criteria, exclusion_criteria):
    table = table.collect()
    flowchart_list = [table.height]
    for crit in inclusion_criteria:
        table = table.filter(pl.col(crit) == True)
        flowchart_list.append(crit)
        flowchart_list.append(table.height)
    for crit in exclusion_criteria:
        table = table.filter(pl.col(crit) == False)
        flowchart_list.append(crit)
        flowchart_list.append(table.height)
    return table.select(
        "Global ICU Stay ID",
        "bicarbonate_exposure",
        "first_bicarbonate_administration_seconds",
        "time_to_bicarbonate_administration_seconds",
        "inclusion_time_seconds_one_lab",
        "exposed_in_24h"
    ), flowchart_list

def get_flowchart(x: list):
    flow = ""
    for i, item in enumerate(x): 
        if i % 2 == 0:
            flow += f"{item}\n"
        else:
            flow += "   |\n"
            flow += f"{item} -----> {x[i-1] - x[i+1]} \n"
            flow += "   |\n"
            flow += "   v\n"
    return flow

LOOSE_ANALYSIS_TABLE, loose_flowchart = filter_analysis_table(ANALYSIS_TABLE, loose_inclusion_criteria, loose_exclusion_criteria)
STRICT_ANALYSIS_TABLE, strict_flowchart = filter_analysis_table(ANALYSIS_TABLE, strict_inclusion_criteria, strict_exclusion_criteria)


if __name__ == "__main__":

    print("Looser Inclusion / Exclusion Criteria Flowchart:")
    print(get_flowchart(loose_flowchart))
    print("Explosed patients in 24h:", LOOSE_ANALYSIS_TABLE.filter(pl.col("exposed_in_24h") == True).height)
    
    print("\n\n---------------------------------\n\n")
    
    print("Stricter Inclusion / Exclusion Criteria Flowchart:")
    print(get_flowchart(strict_flowchart))
    print("Explosed patients in 24h:", STRICT_ANALYSIS_TABLE.filter(pl.col("exposed_in_24h") == True).height)


