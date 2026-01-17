# Bicarb ICU 1 Target Trial emulation 

A quick python script to emulate the proposed research protocol. 
Inclusion criteria 
- >= 18yo
- Severe acidemia within 48h ouf ICU admission
- SOFA >= 4
- Lactate >= 2

Exclusion criteria
- Respiratory acidosis
- Ketoacidosis ( Working definition > 3)
- Volume Loss > 1,5L ( Currently not implemented )
- RRT prior to acidosis
- CKD stages 4, 5 or AKI or GFR<30 at time of acidosis ( eGFR probably not implemented since not imputable on the private Charité dataset)

## Usage 
To generate the criteria_parquet using the build_inclusion_exclusion.py file, this should create a DF with all the inclusion criteria and true false values for a given inclusion criteria. 
TODO : CHANGE TO IMPLEMENT ANALYSIS DF Import the biuild_analysis_df.py to build the necessary structures for analysis.
Use analyse.py to perform analysis. 

## Current problems 
- Volume Intake, Outtake function are unclear in inclusion 
- Inclusion time is defined on any lab value not on current based on chosen model. 

## Current Cohort Flowchart
see inclusion_exclusion ipynb

## TODO : 
- SOFA and eGFG data calculated based on "homemade" model, not the reprodICU standard. (doesn't compile on my computer)

TODO Maxime : 
- Flowchart 
- Amsterdam 
- Relance Physio 
- DL MARS

Changes Moritz : 
- acidoses same row 
- ketones > 4 
- Volume loss !!= 
- not eGFR ? 
-  

