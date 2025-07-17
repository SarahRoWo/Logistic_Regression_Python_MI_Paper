# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook for Step 2a: Obtain ICD and SNOMED diagnoses for outliers

# COMMAND ----------

import pandas as pd
from MI_Functions import *

# From https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html:
# Enable Arrow-based columnar data transfers
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent demographics' files

# COMMAND ----------

mi_demo = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")
vas_only_demo = pd.read_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

all_demo = pd.concat([mi_demo, vas_only_demo])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Read in outlier patients' demographics

# COMMAND ----------

# unstratified patients
demo_outlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_outlier.pkl")

# male infertility patients
mi_demo_outlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_outlier_mi.pkl")

# vasectomy patients
vas_only_demo_outlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_outlier_vas.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in condition_occurrence and concept tables for patients

# COMMAND ----------

# Conditions
pt_cond_case = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_case.csv", sep="\t")
pt_cond_control = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_control.csv", sep="\t")
pt_cond= pd.concat([pt_cond_case, pt_cond_control], ignore_index=True)


pt_cond.rename(columns={"condition_start_DATE": "condition_start_date"},inplace=True)

# Concepts
concepts = pd.read_csv("male_infertility_validation/raw_data/concepts.csv")
# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain SNOMED and corresponding ICD diagnoses for outliers

# COMMAND ----------

# Unstratified patients
diag_outlier = obtain_icd_snomed_diag(demo_df=demo_outlier, pt_cond=pt_cond, concepts=concepts)
print(f"Number of diag_outlier patients: {diag_outlier['person_id'].nunique()}")

# Patients with male infertility
mi_diag_outlier = obtain_icd_snomed_diag(demo_df=mi_demo_outlier, pt_cond=pt_cond, concepts=concepts)
print(f"Number of mi_diag_outlier patients: {mi_diag_outlier['person_id'].nunique()}")

# Vasectomy only patients
vas_only_diag_outlier = obtain_icd_snomed_diag(demo_df=vas_only_demo_outlier, 
                                       pt_cond=pt_cond, 
                                       concepts=concepts)
print(f"Number of vas_only_diag_outlier patients: {vas_only_diag_outlier['person_id'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge analysis_cutoff_date to diagnoses dataframes for outliers

# COMMAND ----------

diag2_outlier = diag_outlier.merge(all_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

mi_diag2_outlier = mi_diag_outlier.merge(mi_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

vas_only_diag2_outlier = vas_only_diag_outlier.merge(vas_only_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add columns specifying whether conditions first began before or after analysis cutoff dates for outliers

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unstratified patients

# COMMAND ----------

diag2_outlier['diag_time_before'] = diag2_outlier['condition_start_date'] < diag2_outlier['analysis_cutoff_date']
diag2_outlier['diag_time_after'] = diag2_outlier['condition_start_date'] > diag2_outlier['analysis_cutoff_date']
diag2_outlier['diag_time_same'] = diag2_outlier['condition_start_date'] == diag2_outlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {diag2_outlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {diag2_outlier[diag2_outlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {diag2_outlier[diag2_outlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as the cutoff date: {diag2_outlier[diag2_outlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Male infertility patients

# COMMAND ----------

mi_diag2_outlier['diag_time_before'] = mi_diag2_outlier['condition_start_date'] < mi_diag2_outlier['analysis_cutoff_date']
mi_diag2_outlier['diag_time_after'] = mi_diag2_outlier['condition_start_date'] > mi_diag2_outlier['analysis_cutoff_date']
mi_diag2_outlier['diag_time_same'] = mi_diag2_outlier['condition_start_date'] == mi_diag2_outlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {mi_diag2_outlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {mi_diag2_outlier[mi_diag2_outlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {mi_diag2_outlier[mi_diag2_outlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as the cutoff date: {mi_diag2_outlier[mi_diag2_outlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vasectomy patients

# COMMAND ----------

vas_only_diag2_outlier['diag_time_before'] = vas_only_diag2_outlier['condition_start_date'] < vas_only_diag2_outlier['analysis_cutoff_date']
vas_only_diag2_outlier['diag_time_after'] = vas_only_diag2_outlier['condition_start_date'] > vas_only_diag2_outlier['analysis_cutoff_date']
vas_only_diag2_outlier['diag_time_same'] = vas_only_diag2_outlier['condition_start_date'] == vas_only_diag2_outlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {vas_only_diag2_outlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {vas_only_diag2_outlier[vas_only_diag2_outlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {vas_only_diag2_outlier[vas_only_diag2_outlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as cutoff date: {vas_only_diag2_outlier[vas_only_diag2_outlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save

# COMMAND ----------

# convert male infertility diagnoses pandas DataFrame back to spark DataFrame
#diag_final_outlier = spark.createDataFrame(diag2_outlier)


diag2_outlier.to_pickle("male_infertility_validation/revision_files/icd_snomed_outlier.pkl")


mi_diag2_outlier.to_pickle("male_infertility_validation/revision_files/mi_icd_snomed_outlier.pkl")


vas_only_diag2_outlier.to_pickle("male_infertility_validation/revision_files/vas_icd_snomed_outlier.pkl")


# COMMAND ----------


