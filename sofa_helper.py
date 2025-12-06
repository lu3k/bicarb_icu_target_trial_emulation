import polars as pl

def calc_sofa_coag(labs):
    return labs.with_columns(
        (
            pl.when(pl.col("Platelets").is_null())
            .then(None)
            .when(pl.col("Platelets").struct.field("value") < 20)
            .then(4)
            .when(pl.col("Platelets").struct.field("value") < 50)
            .then(3)
            .when(pl.col("Platelets").struct.field("value") < 100)
            .then(2)
            .when(pl.col("Platelets").struct.field("value") < 150)
            .then(1)
            .otherwise(0)
        ).alias("sofa_coag")
    )

def calc_sofa_liver(labs):
    return labs.with_columns(
        (
            pl.when(pl.col("Bilirubin").is_null())
            .then(None)
            .when(pl.col("Bilirubin").struct.field("value") >= 12.0)
            .then(4)
            .when(pl.col("Bilirubin").struct.field("value") >= 6.0)
            .then(3)
            .when(pl.col("Bilirubin").struct.field("value") >= 2.0)
            .then(2)
            .when(pl.col("Bilirubin").struct.field("value") >= 1.2)
            .then(1)
            .otherwise(0)
        ).alias("sofa_liver")
    )

def calc_sofa_renal(labs):
    return labs.with_columns(
        (
            pl.when(pl.col("Creatinine").is_null())
            .then(None)
            .when(pl.col("Creatinine").struct.field("value") >= 5.0)
            .then(4)
            .when(pl.col("Creatinine").struct.field("value") >= 3.5)
            .then(3)
            .when(pl.col("Creatinine").struct.field("value") >= 2.0)
            .then(2)
            .when(pl.col("Creatinine").struct.field("value") >= 1.2)
            .then(1)
            .otherwise(0)
        ).alias("sofa_renal")
    )

def calc_sofa_respiratory(labs, respiratory):
    fio2_mask = (
        pl.when(pl.col("FiO2_inhaled").is_not_null()).then(pl.col("FiO2_inhaled"))
        .when(pl.col("FiO2_ventilator").is_not_null()).then(pl.col("FiO2_ventilator"))
        .when(pl.col("FiO2_ventilator").is_null() & pl.col("FiO2_inhaled").is_null() & pl.col("Oxygen gas flow Oxygen delivery system").is_not_null())
        .then(pl.lit(21) + pl.col("Oxygen gas flow Oxygen delivery system")*4)
        .otherwise(21)
    ).alias("FiO2")

    fio2_data = respiratory.select(
        pl.col("Global ICU Stay ID").alias("id"), 
        pl.col("Time Relative to Admission (seconds)").alias("time"), 
        pl.col("Oxygen gas flow Oxygen delivery system"),
        pl.col("Oxygen/Gas total [Pure volume fraction] Inhaled gas").alias("FiO2_inhaled"),
        pl.col("Oxygen/Total gas setting [Volume Fraction] Ventilator").alias("FiO2_ventilator")
    ).with_columns(fio2_mask)

    fio2_data = fio2_data.join_asof(
        labs.select(
            pl.col("Global ICU Stay ID").alias("id"), 
            pl.col("Time Relative to Admission (seconds)").alias("time"),
            pl.col("Oxygen").alias("PaO2")
        ).filter(pl.col("PaO2").is_not_null()),
        on="time",
        by="id",
        strategy="backward",
        tolerance=3600
    )

    pao2_fio2_mask = (
        pl.when(pl.col("PaO2").is_not_null() & pl.col("FiO2").is_not_null())
        .then(pl.col("PaO2").struct.field("value") / pl.col("FiO2"))
        .otherwise(None)
    ).alias("pao2_fio2")
    fio2_data = fio2_data.with_columns(pao2_fio2_mask)

    resp_score = (
        pl.when(pl.col("pao2_fio2").is_null()).then(None)
        .when(pl.col("pao2_fio2") >= 400).then(0)
        .when(pl.col("pao2_fio2") < 400).then(1)
        .when(pl.col("pao2_fio2") < 300).then(2)
        .when((pl.col("pao2_fio2") < 200) & pl.col("FiO2_ventilator").is_not_null()).then(3)
        .when((pl.col("pao2_fio2") < 100) & pl.col("FiO2_ventilator").is_not_null()).then(4)
        .otherwise(0)
    ).alias("sofa_resp")

    #x = fio2_data.filter(pl.col("PaO2").is_not_null() & pl.col("FiO2").is_not_null())
    #not_both_values = fio2_data.select(pl.col("id")).unique().join(x.select(pl.col("id")).unique(), on="id", how="anti")
    return fio2_data.with_columns(resp_score)

def calc_sofa_cardio(vitals, meds, patients):
    # Cardiovascular SOFA score calculation needs MAP and medication

    ## MAP CALCULATION : Uses invasive / non invasive MAP if available
    ## Otherwise calculates MAP from systolic and diastolic pressures DBP - 1/3 (SBP - DBP)
    MAP = (
        pl.when(pl.col("Invasive mean arterial pressure").is_not_null())
        .then(pl.col("Invasive mean arterial pressure"))
        .when(pl.col("Non-invasive mean arterial pressure").is_not_null())
        .then(pl.col("Non-invasive mean arterial pressure"))
        .when(
            pl.col("Invasive mean arterial pressure").is_null() & 
            pl.col("Non-invasive mean arterial pressure").is_null() & 
            pl.col("Non-invasive diastolic arterial pressure").is_not_null() & 
            pl.col("Non-invasive systolic arterial pressure").is_not_null())
            .then(
                (pl.col("Non-invasive diastolic arterial pressure") + 
                    1/3 * (
                        pl.col("Non-invasive systolic arterial pressure") - 
                        pl.col("Non-invasive diastolic arterial pressure")
                    ) 
                )
            )
        .when(
            pl.col("Invasive mean arterial pressure").is_null() & 
            pl.col("Non-invasive mean arterial pressure").is_null() & 
            pl.col("Invasive diastolic arterial pressure").is_not_null() & 
            pl.col("Invasive systolic arterial pressure").is_not_null())
            .then(
                (pl.col("Invasive diastolic arterial pressure") + 
                    1/3 * (
                        pl.col("Invasive systolic arterial pressure") - 
                        pl.col("Invasive diastolic arterial pressure")
                    ) 
                )
            )
        .otherwise(None)
    ).alias("MAP")
    vitals = vitals.with_columns(MAP)

    ## Tranform meds into usable mcg/kg/min dosages 
    ## SOFA scores only apply on continous IV drips
    selected_meds = meds.rename({
        "Global ICU Stay ID": "id"}).filter(
            pl.col("Drug Ingredient").is_in(["dopamine", "dobutamine", "epinephrine", "norepinephrine",])
    ).filter(pl.col("Drug Administration Route").str.contains("intravenous") & (pl.col("Drug is Continuous Infusion") == True))

    selected_meds = selected_meds.join(
        patients.select(
            pl.col("Global ICU Stay ID").alias("id"),
            pl.col("Admission Weight (kg)").alias("admission_weight_kg")
        ),on="id"
    ).rename({
        "Drug Start Relative to Admission (seconds)": "start_time",
        "Drug End Relative to Admission (seconds)": "end_time"
    })


    drug_rate_mcg_min_kg_imputer = (
        pl.when(pl.col("Drug Rate").is_not_null() & (pl.col("Drug Rate Unit") == "mcg/min") & pl.col("admission_weight_kg").is_not_null())
        .then(pl.col("Drug Rate") / pl.col("admission_weight_kg"))
        .when(pl.col("Drug Rate").is_not_null() & (pl.col("Drug Rate Unit") == "mcg/kg/min"))
        .then(pl.col("Drug Rate"))
        .when(pl.col("Drug Rate").is_not_null() & (pl.col("Drug Rate Unit") == "mg/kg/min"))
        .then(pl.col("Drug Rate") * 1000)
        .otherwise(0)
    ).alias("drug_rate_mcg_min_kg")
    selected_meds = selected_meds.with_columns(drug_rate_mcg_min_kg_imputer)

    # Can't impute dosage for patients having only a ml/hr rate (no dosage !!) - solution ?? 
    # iv_drugs.filter(pl.col("drug_rate_mcg_min_kg").is_null() & pl.col("admission_weight_kg").is_not_null() & (pl.col("Drug Rate Unit") == "ml/hr")).collect()

    ## DIRP DRUGS : Pivot medication dataframe to have separate columns for each drug ingredient and calculate cumulative sum over time
    pivot_meds = (
        selected_meds.collect()
        .rename({"start_time": "time"})
        .pivot(
            values="drug_rate_mcg_min_kg",
            index=["id", "time"],
            columns="Drug Ingredient",
            aggregate_function="mean",
        )
        .sort(["id", "time"])
    )

    ## Combine vitals and medication data to calculate cardiovascular SOFA score
    cardio_df = vitals.select(
        pl.col("Global ICU Stay ID").alias("id"),
        pl.col("Time Relative to Admission (seconds)").alias("time"),
        pl.col("MAP"),
    ).join_asof(
        pivot_meds.lazy(),
        on="time",
        by="id",
        strategy="backward",
        tolerance=3600  # 1 hour tolerance
    )

    cardio_sofa = ( ## medicaton values in mcg/kg/min
        pl.when(pl.col("MAP").is_null() & pl.col("dobutamine").is_null() & pl.col("dopamine").is_null() & pl.col("norepinephrine").is_null() & pl.col("epinephrine").is_null())
        .then(None)
        .when(pl.col("MAP").is_not_null() & (pl.col("MAP") >= 70))
        .then(0)
        .when((pl.col("MAP").is_not_null() & (pl.col("MAP") < 70)))
            .then(1)
        .when(
            (pl.col("dobutamine").is_not_null()) |
            (pl.col("dopamine").is_not_null() & (pl.col("dopamine") <= 5)) 
        )
            .then(2)
        .when(
            (pl.col("dopamine").is_not_null() & (pl.col("dopamine") > 5)) |
            (pl.col("norepinephrine").is_not_null() & (pl.col("norepinephrine") <= 0.1)) |
            (pl.col("epinephrine").is_not_null() & (pl.col("epinephrine") <= 0.1))
        )
            .then(3)
        .when(
            (pl.col("dopamine").is_not_null() & (pl.col("dopamine") > 15)) |
            (pl.col("norepinephrine").is_not_null() & (pl.col("norepinephrine") > 0.1)) |
            (pl.col("epinephrine").is_not_null() & (pl.col("epinephrine") > 0.1))
        ) 
            .then(4)
        .otherwise(None)
    ).alias("sofa_cardio")
    return cardio_df.with_columns(cardio_sofa)

def calc_sofa(patients:pl.LazyFrame, vitals:pl.LazyFrame, meds:pl.LazyFrame, labs:pl.LazyFrame, respiratory:pl.LazyFrame) -> pl.LazyFrame:
    result = labs.rename({"Global ICU Stay ID": "id", "Time Relative to Admission (seconds)":"time"})
    result = calc_sofa_coag(result)
    result = calc_sofa_liver(result)
    result = calc_sofa_renal(result).select(
        pl.col("id"),
        pl.col("time"),
        pl.col("sofa_coag"),
        pl.col("sofa_liver"),
        pl.col("sofa_renal")
    )
    return result.join_asof(
            calc_sofa_cardio(vitals, meds, patients).select(
                pl.col("id"), 
                pl.col("time"),
                pl.col("sofa_cardio")
            ),
            on="time",
            by="id",
            strategy="backward",
            tolerance=3600
        ).join_asof(
            calc_sofa_respiratory(labs, respiratory).select(
                pl.col("id"), 
                pl.col("time"),
                pl.col("sofa_resp")
            ),
            on="time",
            by="id",
            strategy="backward",
            tolerance=3600
        ).with_columns(
            pl.sum_horizontal(["sofa_coag", "sofa_liver", "sofa_renal", "sofa_cardio", "sofa_resp"]).alias("sofa")
        )

