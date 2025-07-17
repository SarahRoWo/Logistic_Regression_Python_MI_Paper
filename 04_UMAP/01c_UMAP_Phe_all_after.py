# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## UMAP for patients with male infertility vs patients without male infertility diagnosis (but can include patients who have had vasectomy)
# MAGIC
# MAGIC ### This creates UMAPs for patients' phenotypic profiles > 6 months after diagnosis/procedure
# MAGIC
# MAGIC Note: This notebook includes all diagnoses

# COMMAND ----------

!pip install numba
!pip install scikit-posthocs
!pip install umap-learn==0.5.3

# COMMAND ----------
runfile('/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI/MI_Functions.py', wdir='/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.core.multiarray
import numpy as np
import os
import umap
import re
import scipy
from scipy import stats
from scipy.stats import mstats
from scikit_posthocs import posthoc_dunn
import matplotlib
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run MI_Functions.py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in demographics and diagnoses files for male infertility and vasectomy patients

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demographics

# COMMAND ----------

demographics = dict()

demos = ['mi',
         'vas_only']
file_names_demos = ['mi_pts_only_final',
                    'vas_pts_only_final']

for demo, file_name_demo in zip(demos, file_names_demos):
  temp = pd.read_pickle(f"male_infertility_validation/demographics/{file_name_demo}.pkl")
  demographics[demo] = temp

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnoses

# COMMAND ----------

diagnoses = dict()

diags = ['mi_diag', 
         'vas_only_diag']
file_names_diags = ['mi_phe', 
                    'vas_phe']

for diag, file_name_diag in zip(diags, file_names_diags):
  temp = pd.read_pickle(f"male_infertility_validation/diagnoses/{file_name_diag}.pkl")
  diagnoses[diag] = temp

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter for phenotypes occurring > 6 months after diagnosis/procedure

# COMMAND ----------

temp = diagnoses['mi_diag'].copy()
temp.columns

# COMMAND ----------

diagnoses_after = dict()

# 'mi_diag'
temp = diagnoses['mi_diag'].copy()
temp = temp[temp['phe_time_after'] == 1]
diagnoses_after['mi_diag'] = temp

# 'vas_only_diag'
temp = diagnoses['vas_only_diag'].copy()
temp = temp[temp['phe_time_after'] == 1]
diagnoses_after['vas_only_diag'] = temp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make pivot table containing all patients' phenotypic profiles

# COMMAND ----------

pivot_tables = dict()

for file_name_diag in diagnoses_after:
  pivot_tables[file_name_diag] = make_pivot_tables(diagnoses_after[file_name_diag], file_name_diag, n='phenotype')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Concatenate pivot tables

# COMMAND ----------

alldiag_pivot = pd.concat([pivot_tables['mi_diag'], pivot_tables['vas_only_diag']], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop infertility-related conditions

# COMMAND ----------

colstodrop = list()

colstodrop.extend(alldiag_pivot.columns[alldiag_pivot.columns.str.contains('infertility|spermia|reproduction',      
                                                                           flags=re.IGNORECASE)])

# remove male infertility status as a column to drop
colstodrop.remove('male infertility status')

colstodrop = set(colstodrop)
print(colstodrop)

# COMMAND ----------

alldiag_pivot = alldiag_pivot.drop(colstodrop, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make demographic dfs

# COMMAND ----------

demographics['mi'].columns

# COMMAND ----------

demographic_cols = ['person_id', 'year_of_birth', 'estimated_age', 'gender', 'race',
                    'ethnicity', #'location_source_value','adi', 'adi_category',
                    'num_visits_after', 'emr_months_after']

# COMMAND ----------

# Only keep demographic columns and merge
demographics['mi'] = demographics['mi'][demographic_cols].copy()
demographics['vas_only'] = demographics['vas_only'][demographic_cols].copy()

all_demo_df = pd.concat([demographics['mi'], demographics['vas_only']], axis=0, copy=False)

# COMMAND ----------

all_demo_df = all_demo_df.drop_duplicates(). \
                          set_index('person_id').reindex(alldiag_pivot.index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check that alldiag_pivot and all_demo dfs have the same number of rows, the same index, and fillna with 0.

# COMMAND ----------

# MAGIC %md
# MAGIC ### fillna with 0

# COMMAND ----------

all_demo_df = all_demo_df.fillna(0)
alldiag_pivot = alldiag_pivot.fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the shape of demo and diag dfs

# COMMAND ----------

print(f"Shape of all_demo_df is {all_demo_df.shape}")
print(f"Shape of alldiag_pivot is {alldiag_pivot.shape}")
print(f"Number of rows of the dataframes are the same: {all_demo_df.shape[0] == alldiag_pivot.shape[0]}")
print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check whether indices are the same for demo and diag dfs

# COMMAND ----------

demo_index = all_demo_df.index
diag_index = alldiag_pivot.index
print(f"indices are the same for all_demo_df and alldiag_pivot: {demo_index.equals(diag_index)}")
print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimensionality Reduction

# COMMAND ----------

alldiag_pivot

# COMMAND ----------

X = dimensionality_reduction(diag=alldiag_pivot, file_name='mi_vas_only_after')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save 'y', which will preserve each patient's male infertility status and other demographic features (it is not preserved after performing dimensionality reduction)

# COMMAND ----------

temp = alldiag_pivot.copy()
y = temp['male infertility status'].replace({1 : 'male infertility patient', 0 : 'control (vasectomy patient)'})
y = y.to_frame()
y = y.reset_index()
y = y.reset_index()
y = y.merge(all_demo_df, on='person_id')
y.to_pickle("male_infertility_validation/tables/umap/y_all_after.pkl")

# COMMAND ----------

y['male infertility status'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save demographic dataframes

# COMMAND ----------

temp = all_demo_df.copy()
temp = temp.reset_index()
temp.to_pickle("male_infertility_validation/tables/umap/demo_all_after.pkl")

# COMMAND ----------

