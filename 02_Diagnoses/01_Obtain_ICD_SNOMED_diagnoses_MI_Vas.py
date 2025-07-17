# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Obtain SNOMED diagnoses. This notebook includes ICD9CM and ICD10CM diagnoses. (Updated 20230511)

# COMMAND ----------

import pandas as pd

# From https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html:
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run MI_Functions.py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent demographics' files

# COMMAND ----------

mi_demo = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")

vas_only_demo = pd.read_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in condition_occurrence and concept tables for patients

# COMMAND ----------

# Conditions
# male infertility patients
#mi_cond= pd.read_pickle("male_infertility_validation/raw_data/condition_occurrence_case.pkl")

mi_cond = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_case.csv", sep="\t")
concepts = pd.read_csv("male_infertility_validation/raw_data/concepts.csv")
vas_cond = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_control.csv",sep="\t")



def convert_columns_to_lowercase(df):
    new_column_names = {col: col.lower() for col in df.columns}
    df.rename(columns=new_column_names, inplace=True)
    return df

mi_cond = convert_columns_to_lowercase(mi_cond)
vas_cond= convert_columns_to_lowercase(vas_cond)
concepts= convert_columns_to_lowercase(concepts)

# vasectomy patients
#vas_cond= pd.read_pickle("male_infertility_validation/raw_data/condition_occurrence_control.pkl")

# Concepts
#concepts= pd.read_pickle("male_infertility_validation/raw_data/concept.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain SNOMED and corresponding ICD diagnoses

# COMMAND ----------

# Patients with paternal infertility
mi_diag = obtain_icd_snomed_diag_stanford(demo_df=mi_demo, 
                                          pt_cond=mi_cond, 
                                          concepts=concepts, 
                                          file_name='mi_icd_snomed')

# Vasectomy only patients
vas_only_diag = obtain_icd_snomed_diag_stanford(demo_df=vas_only_demo, 
                                                pt_cond=vas_cond, 
                                                concepts=concepts,
                                                file_name='vas_only_icd_snomed')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge analysis_cutoff_date to diagnoses dataframes

# COMMAND ----------

mi_diag2 = mi_diag.merge(mi_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

vas_only_diag2 = vas_only_diag.merge(vas_only_demo[['person_id', 'analysis_cutoff_date']], on='person_id', how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add columns specifying whether conditions first began before or after analysis cutoff dates

# COMMAND ----------

# MAGIC %md
# MAGIC #### Male infertility patients

# COMMAND ----------

mi_diag2['diag_time_before'] = mi_diag2['condition_start_date'] < mi_diag2['analysis_cutoff_date']
mi_diag2['diag_time_after'] = mi_diag2['condition_start_date'] > mi_diag2['analysis_cutoff_date']
mi_diag2['diag_time_same'] = mi_diag2['condition_start_date'] == mi_diag2['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {mi_diag2.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {mi_diag2[mi_diag2['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {mi_diag2[mi_diag2['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as the cutoff date: {mi_diag2[mi_diag2['diag_time_same']==True].shape[0]}")

#Total number of rows: 143066
#Number of rows where condition started before cutoff date: 63679
#Number of rows where condition started after cutoff date: 79328
#Number of rows where condition started at the same time as the cutoff date: 59


# COMMAND ----------

mi_diag_before = mi_diag2[mi_diag2['diag_time_before']==True]
print(color.BOLD + "Top 10 diagnoses before cutoff date:" + color.END)
#display(mi_diag_before['icd_concept_name'].value_counts().head(10))

mi_diag_after = mi_diag2[mi_diag2['diag_time_after']==True]
print(color.BOLD + "\nTop 10 diagnoses after cutoff date:" + color.END)
#display(mi_diag_after['icd_concept_name'].value_counts().head(10))

mi_diag_same = mi_diag2[mi_diag2['diag_time_same']==True]
print(color.BOLD + "\nTop 10 diagnoses that occur at the same time as the cutoff date:" + color.END)
#display(mi_diag_same['icd_concept_name'].value_counts().head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vasectomy patients

# COMMAND ----------

vas_only_diag2['diag_time_before'] = vas_only_diag2['condition_start_date'] < vas_only_diag2['analysis_cutoff_date']
vas_only_diag2['diag_time_after'] = vas_only_diag2['condition_start_date'] > vas_only_diag2['analysis_cutoff_date']
vas_only_diag2['diag_time_same'] = vas_only_diag2['condition_start_date'] == vas_only_diag2['analysis_cutoff_date']

# COMMAND ----------

print(f"Total number of rows: {vas_only_diag2.shape[0]}")
print(f"Number of rows where condition started before cutoff date: {vas_only_diag2[vas_only_diag2['diag_time_before']==True].shape[0]}")
print(f"Number of rows where condition started after cutoff date: {vas_only_diag2[vas_only_diag2['diag_time_after']==True].shape[0]}")
print(f"Number of rows where condition started at the same time as cutoff date: {vas_only_diag2[vas_only_diag2['diag_time_same']==True].shape[0]}")

#Total number of rows: 65126
#Number of rows where condition started before cutoff date: 37635
#Number of rows where condition started after cutoff date: 27474
#Number of rows where condition started at the same time as cutoff date: 17

# COMMAND ----------

vas_only_diag_before = vas_only_diag2[vas_only_diag2['diag_time_before']==True]
print(color.BOLD + "Top 10 diagnoses before cutoff date:" + color.END)
#display(vas_only_diag_before['icd_concept_name'].value_counts().head(10))

vas_only_diag_after = vas_only_diag2[vas_only_diag2['diag_time_after']==True]
print(color.BOLD + "\nTop 10 diagnoses after cutoff date:" + color.END)
#display(vas_only_diag_after['icd_concept_name'].value_counts().head(10))

vas_only_diag_same = vas_only_diag2[vas_only_diag2['diag_time_same']==True]
print(color.BOLD + "\nTop 10 diagnoses that occur at the same time as cutoff date" + color.END)
#display(vas_only_diag_same['icd_concept_name'].value_counts().head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save diagnoses for male infertility and vasectomy patients

# COMMAND ----------

# save male infertility patients' diagnoses
mi_diag2.to_pickle("male_infertility_validation/diagnoses/mi_icd_snomed.pkl")

# save vasectomy patients' diagnoses
vas_only_diag2.to_pickle("male_infertility_validation/diagnoses/vas_icd_snomed.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check SNOMED diagnoses

# COMMAND ----------

# MAGIC %md
# MAGIC patients with male infertility

# COMMAND ----------

check_icd_snomed_diag_stanford(diag_pts=mi_diag2, demo_pts=mi_demo)
#"['condition_source_value'] not in index"

# COMMAND ----------

# MAGIC %md
# MAGIC vasectomy patients

# COMMAND ----------

check_icd_snomed_diag_stanford(diag_pts=vas_only_diag2, demo_pts=vas_only_demo)

# COMMAND ----------

