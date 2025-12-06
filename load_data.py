import reprodICU

print(reprodICU.__version__)

datasets=["nwICU", "eICU", "MIMIC3", "MIMIC4"]

#reprodICU.build.build_patient_information(datasets=["eICU", "MIMIC3", "MIMIC4"])
# ALL 
#reprodICU.build.build_all(datasets=datasets, demo=False)

# Patient demographics
#reprodICU.build.build_patient_information(datasets=datasets)

# Diagnostic codes
#reprodICU.build.build_diagnoses(datasets=datasets)

# Procedures and interventions
#reprodICU.build.build_procedures(datasets=datasets)

# Medication records
#reprodICU.build.build_medications(datasets=datasets)

# Microbiology cultures
#reprodICU.build.build_microbiology(datasets=datasets)

# Clinical notes
#reprodICU.build.build_notes(datasets=datasets)


## à faire : 
# Timeseries data (vitals, labs, respiratory, intake/output)
#reprodICU.build.build_timeseries(
#    datasets=datasets,
#    timeseries=["vitals", "labs", "respiratory", "inout"],
#)

# Derived clinical concepts
reprodICU.build.build_magic_concepts(
    datasets=datasets,
    concepts=["RENAL_REPLACEMENT_THERAPY"]
)