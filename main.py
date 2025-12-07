import build_analysis_df
import analyse

km = build_analysis_df.get_kaplan_meier_pd()
analyse.kaplan_meier(km)