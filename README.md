# Bicarb ICU 1 Target Trial emulation 

A quick python script to emulate the proposed research protocol. 
Inclusion criteria 
- >= 18yo
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
- Problem with ReprodICU where there are lots of labs with no patient_info and vice versa. See undefined_labs_problem_analysis.ipynb for details. 

## Current Cohort Flowchart
356,849 Patients in Datasets<br>
-> 8809 <18yo<br>
-> 346,736 with no acidosis<br> 
-> 279 with SOFA <4<br>
-> 99 with Lactate <2<br>
926 Patiens included<br>
-> 611 with respiratoy acidosis<br>
-> 0 with ketosis<br>
-> 69 with >1.5L Volume loss<br>
-> 0 Patients with prior RRT <br>
-> 0 Patients with CKD, 5 Patiens with eGFR<30<br>
241 Patients used in analysis <br>

## TODO : 
- cohort size very small after selection of acidosis criteria ? is there an error ?
- SOFA and eGFG data calculated based on "homemade" model, not the reprodICU standard. (doesn't compile on my computer)
- Rebuild as original study 

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

