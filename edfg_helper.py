import polars as pl

def eDFG_ckd_epi(patient_info: pl.LazyFrame, labs: pl.LazyFrame) -> pl.LazyFrame:
    eDFG = patient_info.select(
        pl.col("Global ICU Stay ID"),
        pl.col("Admission Age (years)"),
        pl.col("Gender"),
        pl.col("Ethnicity")
    ).join(
        labs.select(
            pl.col("Global ICU Stay ID"),
            pl.col("Creatinine") ,
            pl.col("Time Relative to Admission (seconds)")
        ).filter(pl.col("Creatinine").is_not_null()),
        on="Global ICU Stay ID",
        how = "inner"
    )

    k_men = 79.6
    k_women = 61.9

    a_men = -0.411
    a_women = -0.329

    eDFG = eDFG.with_columns(
        pl.when(pl.col("Gender") == "Male")
          .then(
              141 * 
              (pl.min_horizontal(pl.col("Creatinine").struct.field("value") / k_men, 1) ** a_men) *
              (pl.max_horizontal(pl.col("Creatinine").struct.field("value") / k_men, 1) ** -1.209) *
              (0.993 ** pl.col("Admission Age (years)"))
          )
        .when(pl.col("Gender") == "Female")
          .then(
              141 * 
              (pl.min_horizontal(pl.col("Creatinine").struct.field("value") / k_women, 1) ** a_women) *
              (pl.max_horizontal(pl.col("Creatinine").struct.field("value") / k_women, 1) ** -1.209) *
              (0.993 ** pl.col("Admission Age (years)")) *
              1.018
          )
        .otherwise(None)
        .alias("eDFG CKD-EPI")
    )

    return eDFG.select(pl.col("Global ICU Stay ID"), pl.col("eDFG CKD-EPI"), pl.col("Time Relative to Admission (seconds)"))





