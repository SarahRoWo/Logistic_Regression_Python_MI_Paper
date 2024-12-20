# Databricks notebook source
# MAGIC %md
# MAGIC ## Obtain deltas between phecode diagnoses and first mi diagnosis (or vas record)

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' phecode notebooks

# COMMAND ----------

phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")

phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent demographics files

# COMMAND ----------

mi_demo = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")

vas_only_demo = pd.read_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent diagnoses files

# COMMAND ----------

# Patients with male infertility
mi_diag = pd.read_pickle("male_infertility_validation/diagnoses/mi_icd_snomed.pkl")

# Vasectomy only patients
vas_only_diag = pd.read_pickle("male_infertility_validation/diagnoses/vas_icd_snomed.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain corresponding phecodes

# COMMAND ----------

phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")
phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

# COMMAND ----------

phecodes_v2_selected = phecodes_v2[['icd', 'phecode', 'phenotype']]
phecodes_cat_v2_selected = phecodes_cat_v2[['phecode']]

phecodes = phecodes_v2_selected.merge(phecodes_cat_v2_selected,
                                           on='phecode',
                                           how='left').copy().drop_duplicates()
# COMMAND ----------

analyses = dict()

# COMMAND ----------

diags = [mi_diag, 
         vas_only_diag]
diag_names = ['mi_diag', 
              'vas_only_diag']

for diag, diag_name in zip(diags, diag_names):
  print(f"Adding phecodes for {diag_name}")
  analyses[diag_name] = obtain_phecodes_stanford(icd_diag=diag, phecodes=phecodes)
  print('\n')

# COMMAND ----------

mi_diag2 = analyses['mi_diag'].toPandas()
vas_only_diag2 = analyses['vas_only_diag'].toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add in first phenotype start date for each patient and whether it was before or after analysis cutoff date

# COMMAND ----------

# patients with male infertility
mi_first_phe_date = add_first_phe_date(df=mi_diag2)
mi_diag3 = add_phe_rel_date(df=mi_diag2, first_phe_date=mi_first_phe_date, cutoff_date='analysis_cutoff_date')

# patients who have underwent a vasectomy related procedure
vas_only_first_phe_date = add_first_phe_date(df=vas_only_diag2)
vas_only_diag3 = add_phe_rel_date(df=vas_only_diag2, first_phe_date=vas_only_first_phe_date, cutoff_date='analysis_cutoff_date')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find delta between first non mi or vas diagnosis date and first mi or vas diagnosis date

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge in first mi or phe date and change to datetime

# COMMAND ----------

# Merge first mi date for mi patients
mi_diag4 = mi_diag3.merge(mi_demo[['person_id', 'first_mi_or_vas_date']], on='person_id', how='left').drop_duplicates().copy()

# Convert first mi date to datetime
mi_diag4['first_mi_or_vas_date'] = pd.to_datetime(mi_diag4['first_mi_or_vas_date'], infer_datetime_format=True)

# Merge first vas date for vas patients
vas_only_diag4 = vas_only_diag3.merge(vas_only_demo[['person_id', 'first_mi_or_vas_date']], on='person_id', how='left').drop_duplicates().copy()

# Convert first vas date to datetime
vas_only_diag4['first_mi_or_vas_date'] = pd.to_datetime(vas_only_diag4['first_mi_or_vas_date'], infer_datetime_format=True)

# COMMAND ----------

# Making sure merged correctly 

print(f"mi_diag3 and mi_diag4 have same number of rows: {mi_diag3.shape[0] == mi_diag4.shape[0]}")
print(f"vas_only_diag3 and vas_only_diag4 have same number of rows: {vas_only_diag3.shape[0] == vas_only_diag4.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find difference between first non mi or vas diagnosis date and first mi or vas diagnosis date in days and months

# COMMAND ----------

# Male infertility patients
mi_diag4['diag_mi_or_vas_delta'] = mi_diag4['first_phe_start_date'] - mi_diag4['first_mi_or_vas_date']
mi_diag4['diag_mi_or_vas_delta_days'] = mi_diag4['diag_mi_or_vas_delta'].dt.days

# Divides by average number of days in a month
# Source:
# https://stackoverflow.com/questions/51918024/python-timedelta64-convert-days-to-months
mi_diag4['diag_mi_or_vas_delta_approx_m'] = mi_diag4['diag_mi_or_vas_delta'] / np.timedelta64(1, 'M')

# Vasectomy patients
vas_only_diag4['diag_mi_or_vas_delta'] = vas_only_diag4['first_phe_start_date'] - vas_only_diag4['first_mi_or_vas_date']
vas_only_diag4['diag_mi_or_vas_delta_days'] = vas_only_diag4['diag_mi_or_vas_delta'].dt.days

# Divides by average number of days in a month
# Source:
# https://stackoverflow.com/questions/51918024/python-timedelta64-convert-days-to-months
vas_only_diag4['diag_mi_or_vas_delta_approx_m'] = vas_only_diag4['diag_mi_or_vas_delta'] / np.timedelta64(1, 'M')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find difference between first non mi or vas diagnosis date and analysis cutoff date in days and months

# COMMAND ----------

### Find difference between first non mi or vas diagnosis date and first mi or vas diagnosis date in days and months
# Male infertility patients
mi_diag4['diag_cutoff_delta'] = mi_diag4['first_phe_start_date'] - mi_diag4['analysis_cutoff_date']
mi_diag4['diag_cutoff_delta_days'] = mi_diag4['diag_cutoff_delta'].dt.days

# Divides by average number of days in a month
# Source:
# https://stackoverflow.com/questions/51918024/python-timedelta64-convert-days-to-months
mi_diag4['diag_cutoff_delta_approx_m'] = mi_diag4['diag_cutoff_delta'] / np.timedelta64(1, 'M')

# Vasectomy patients
vas_only_diag4['diag_cutoff_delta'] = vas_only_diag4['first_phe_start_date'] - vas_only_diag4['analysis_cutoff_date']
vas_only_diag4['diag_cutoff_delta_days'] = vas_only_diag4['diag_cutoff_delta'].dt.days

# Divides by average number of days in a month
# Source:
# https://stackoverflow.com/questions/51918024/python-timedelta64-convert-days-to-months
vas_only_diag4['diag_cutoff_delta_approx_m'] = vas_only_diag4['diag_cutoff_delta'] / np.timedelta64(1, 'M')
vas_only_diag4.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save as pickle

# COMMAND ----------

mi_diag4.to_pickle("male_infertility_validation/revision_files/mi_phe_w_diag_deltas.pkl")
vas_only_diag4.to_pickle("male_infertility_validation/revision_files/vas_phe_w_diag_deltas.pkl")

# COMMAND ----------

