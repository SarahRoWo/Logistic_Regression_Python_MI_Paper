# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook for Step 2b: Obtain ICD and SNOMED diagnoses for nonoutliers

# COMMAND ----------

import pandas as pd
from MI_Functions import *

# From https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html:

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
# MAGIC ## Read in nonoutlier patients' demographics

# COMMAND ----------


# unstratified patients
demo_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_nonoutlier.pkl")

# male infertility patients
mi_demo_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_nonoutlier_mi.pkl")

# vasectomy patients
vas_only_demo_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/Xy_nonoutlier_vas.pkl")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in condition_occurrence and concept tables for patients

# COMMAND ----------

# Conditions
pt_cond_case = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_case.csv", sep="\t")
pt_cond_control = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_control.csv", sep="\t")
pt_cond= pd.concat([pt_cond_case, pt_cond_control], ignore_index=True)

pt_cond.rename(columns={"condition_start_DATE": "condition_start_date"}, inplace=True)

# Concepts
concepts = pd.read_csv("male_infertility_validation/raw_data/concepts.csv")
# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain SNOMED and corresponding ICD diagnoses for nonoutliers

# COMMAND ----------

# Unstratified patients
diag_nonoutlier = obtain_icd_snomed_diag(demo_df=demo_nonoutlier, pt_cond=pt_cond, concepts=concepts)
print(f"Number of diag_nonoutlier patients: {diag_nonoutlier['person_id'].nunique()}")

# Patients with male infertility
mi_diag_nonoutlier = obtain_icd_snomed_diag(demo_df=mi_demo_nonoutlier, pt_cond=pt_cond, concepts=concepts)
print(f"Number of mi_diag_nonoutlier patients: {mi_diag_nonoutlier['person_id'].nunique()}")

# Vasectomy only patients
vas_only_diag_nonoutlier = obtain_icd_snomed_diag(demo_df=vas_only_demo_nonoutlier, 
                                       pt_cond=pt_cond, 
                                       concepts=concepts)
print(f"Number of vas_only_diag_nonoutlier patients: {vas_only_diag_nonoutlier['person_id'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge analysis_cutoff_date to diagnoses dataframes for nonoutliers

# COMMAND ----------

diag2_nonoutlier = diag_nonoutlier.merge(all_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

mi_diag2_nonoutlier = mi_diag_nonoutlier.merge(mi_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

vas_only_diag2_nonoutlier = vas_only_diag_nonoutlier.merge(vas_only_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add columns specifying whether conditions first began before or after analysis cutoff dates for nonoutliers

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unstratified patients

# COMMAND ----------

diag2_nonoutlier['diag_time_before'] = diag2_nonoutlier['condition_start_date'] < diag2_nonoutlier['analysis_cutoff_date']
diag2_nonoutlier['diag_time_after'] = diag2_nonoutlier['condition_start_date'] > diag2_nonoutlier['analysis_cutoff_date']
diag2_nonoutlier['diag_time_same'] = diag2_nonoutlier['condition_start_date'] == diag2_nonoutlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {diag2_nonoutlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {diag2_nonoutlier[diag2_nonoutlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {diag2_nonoutlier[diag2_nonoutlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as the cutoff date: {diag2_nonoutlier[diag2_nonoutlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Male infertility patients

# COMMAND ----------

mi_diag2_nonoutlier['diag_time_before'] = mi_diag2_nonoutlier['condition_start_date'] < mi_diag2_nonoutlier['analysis_cutoff_date']
mi_diag2_nonoutlier['diag_time_after'] = mi_diag2_nonoutlier['condition_start_date'] > mi_diag2_nonoutlier['analysis_cutoff_date']
mi_diag2_nonoutlier['diag_time_same'] = mi_diag2_nonoutlier['condition_start_date'] == mi_diag2_nonoutlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {mi_diag2_nonoutlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {mi_diag2_nonoutlier[mi_diag2_nonoutlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {mi_diag2_nonoutlier[mi_diag2_nonoutlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as the cutoff date: {mi_diag2_nonoutlier[mi_diag2_nonoutlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vasectomy patients

# COMMAND ----------

vas_only_diag2_nonoutlier['diag_time_before'] = vas_only_diag2_nonoutlier['condition_start_date'] < vas_only_diag2_nonoutlier['analysis_cutoff_date']
vas_only_diag2_nonoutlier['diag_time_after'] = vas_only_diag2_nonoutlier['condition_start_date'] > vas_only_diag2_nonoutlier['analysis_cutoff_date']
vas_only_diag2_nonoutlier['diag_time_same'] = vas_only_diag2_nonoutlier['condition_start_date'] == vas_only_diag2_nonoutlier['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {vas_only_diag2_nonoutlier.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {vas_only_diag2_nonoutlier[vas_only_diag2_nonoutlier['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {vas_only_diag2_nonoutlier[vas_only_diag2_nonoutlier['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as cutoff date: {vas_only_diag2_nonoutlier[vas_only_diag2_nonoutlier['diag_time_same']==True].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save

# COMMAND ----------


diag2_nonoutlier.to_pickle("male_infertility_validation/revision_files/icd_snomed_nonoutlier.pkl")


mi_diag2_nonoutlier.to_pickle("male_infertility_validation/revision_files/mi_icd_snomed_nonoutlier.pkl")


vas_only_diag2_nonoutlier.to_pickle("male_infertility_validation/revision_files/vas_icd_snomed_nonoutlier.pkl")


# COMMAND ----------


