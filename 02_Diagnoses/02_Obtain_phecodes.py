# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Obtain phecode diagnoses. This notebook converts icd diagnoses to phecodes. (Updated 20230510)

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
# MAGIC ## 'Import' phecode notebooks

# COMMAND ----------

phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")

phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent diagnoses files

# COMMAND ----------

# Patients with paternal infertility
mi_diag = pd.read_pickle("male_infertility_validation/diagnoses/mi_icd_snomed.pkl")

# Vasectomy only patients
vas_only_diag = pd.read_pickle("male_infertility_validation/diagnoses/vas_icd_snomed.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain corresponding phecodes

# COMMAND ----------

phecodes_v2_selected = phecodes_v2[['icd', 'phecode', 'phenotype']]
phecodes_cat_v2_selected = phecodes_cat_v2[['phecode']]

# COMMAND ----------

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

mi_diag2 = analyses['mi_diag'].copy()
vas_only_diag2 = analyses['vas_only_diag'].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add in first phenotype start date for each patient and whether it was before or after analysis cutoff date

# COMMAND ----------

# patients with paternal infertility
mi_first_phe_date = add_first_phe_date(df=mi_diag2)
mi_diag2_final = add_phe_rel_date(df=mi_diag2, first_phe_date=mi_first_phe_date, cutoff_date='analysis_cutoff_date')

# patients who have underwent a vasectomy related procedure
vas_only_first_phe_date = add_first_phe_date(df=vas_only_diag2)
vas_only_diag2_final = add_phe_rel_date(df=vas_only_diag2, first_phe_date=vas_only_first_phe_date, cutoff_date='analysis_cutoff_date')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save

# COMMAND ----------

mi_diag2_final.to_pickle("male_infertility_validation/diagnoses/mi_phe.pkl")
vas_only_diag2_final.to_pickle("male_infertility_validation/diagnoses/vas_phe.pkl")

# COMMAND ----------

