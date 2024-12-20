# Databricks notebook source
# MAGIC %md
# MAGIC ## Obtain patients lost to followup for each cutoff time (for the logistic regression analyses after the 6-month cutoff with 12, 24 36, 48, or 60 months used as cutoff time)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in pertinent demographics files

# COMMAND ----------

mi_demo = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")

vas_only_demo = pd.read_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Obtain number of male infertility and vasectomy patients

# COMMAND ----------

mi_pts_n = mi_demo['person_id'].nunique()
vas_only_pts_n = vas_only_demo['person_id'].nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make dataframe specifying number of male infertility and vasectomy patients lost to follow-up for each cutoff time

# COMMAND ----------

cutoff_dict = dict()
cutoff_dict['cutoff_times'] = [12, 24, 36, 48, 60]

# COMMAND ----------

cutoff_dict['mi_pts_lost'] = []

for cutoff_time in cutoff_dict['cutoff_times']:
  # Make temp table of patients meeting cutoff threshold
  # Patients who have at least x months of cutoff time
  temp = mi_demo[mi_demo['emr_months_after'] >= cutoff_time].copy()
  temp['mi_pts_lost'] = mi_pts_n - temp['person_id'].nunique()
  cutoff_dict['mi_pts_lost'].append(temp['mi_pts_lost'].values[0])

# COMMAND ----------

cutoff_dict['vas_pts_lost'] = []

for cutoff_time in cutoff_dict['cutoff_times']:
  # Make temp table of patients meeting cutoff threshold
  # Patients who have at least x months of cutoff time
  temp = vas_only_demo[vas_only_demo['emr_months_after'] >= cutoff_time].copy()
  temp['vas_pts_lost'] = vas_only_pts_n - temp['person_id'].nunique()
  cutoff_dict['vas_pts_lost'].append(temp['vas_pts_lost'].values[0])

# COMMAND ----------

pd.DataFrame(cutoff_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save

# COMMAND ----------

cutoff_df = pd.DataFrame(cutoff_dict)

# COMMAND ----------

cutoff_df.csv("male_infertility_validation/revision_files/fx_cutoff.csv")

# COMMAND ----------

