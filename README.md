# Bicarb ICU 1 Target Trial emulation 

A quick python script to emulate the proposed research protocol. 
Inclusion criteria 
- >18yo
- Severe acidemia within 48h ouf ICU admission
- SOFA >= 4
- Lactate >= 2

Exclusion criteria
- Respiratory acidosis
- Ketoacidosis
- Volume Loss > 1,5L
- RRT prior to acidosis
- CKD stages 4, 5 or AKI or GFR<30 at time of acidosis

## Usage 
To generate the cohort run the bicarb_build_cohort.py file, this should build the cohort in the bicarb_data directory. 
Import the biuild_analysis_df.py to build the necessary structures for analysis.
Use analyse.py to perform analysis. 

## Current problems 
- The SOFA and Lactate inclusion criteria are used at any point in time not 48h prior to inclusion event (acidosis) opposed to the proposed research protocol
- Ketones do not have a defined value
- Volume Intake, Outtake function are unclear in inclusion 
- RRT in diagnosis Frame do not have a Global ICU Stay ID. Imputing them with the Global Hospital Stay ID doesn't include all patients / shouldn't be a problem, since all patients with a ICU stay ID that fall into this case are eliminated.
-  

## TODO : 
- cohort size very small after selection of acidosis criteria ? is there an error ?
- SOFA and eGFG data calculated based on "homemade" model, not the reprodICU standard. (doesn't compile on my computer)
- Rebuild as original study 
