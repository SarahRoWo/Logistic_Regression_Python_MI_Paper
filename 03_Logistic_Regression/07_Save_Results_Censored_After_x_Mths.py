# Databricks notebook source
# MAGIC %md
# MAGIC ## Save censored logistic regression analyses 
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This includes the following logistic regression analyses
# MAGIC
# MAGIC - > 6 months after diagnosis/procedure:
# MAGIC 1. `has_mi ~ has_phenotype` 
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_after + months_in_EMR_after` 
# MAGIC
# MAGIC For 12, 24, 36, 48, and 60 month cutoffs; total number of analyses = 20

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import chi2_contingency
import scipy.stats as stats
from math import log10, log2
#from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

# COMMAND ----------

pd.set_option('display.max_rows', 50)
np.set_printoptions(threshold=50)

# COMMAND ----------

diagkeys = ['phenotype']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in logistic regression analysis files

# COMMAND ----------

# MAGIC %md
# MAGIC ### > 6 months after diagnosis/procedure

# COMMAND ----------

analyses_after = dict()

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

cutoff_times = [12, 24, 36, 48, 60]

for cutoff_time in cutoff_times:
  analyses_after[cutoff_time] = dict()
  for file_name in file_names:
    analyses_after[cutoff_time][file_name] = pd.read_pickle(f"male_infertility_validation/revision_files/fx_{file_name}_{str(cutoff_time)}m_cutoff.pkl")

# COMMAND ----------

for cutoff_time in analyses_after:
    for file_name in file_names:
        print(f"Phenotypes significant for patients with male infertility for {cutoff_time} month cutoff, {file_name} analysis")
        temp = analyses_after[cutoff_time][file_name].copy()
        display(temp[temp['significance_bh'] == 'mi_significant'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Censor low counts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Columns that will be censored

# COMMAND ----------

count_cols = analyses_after[12]['primary'].filter(regex='Count')
count_cols.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Censoring > 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

analyses_after_censored = dict()

for cutoff_time in analyses_after:
  analyses_after_censored[cutoff_time] = dict()
  for file_name in file_names:
    temp = analyses_after[cutoff_time][file_name].copy()

    for col in count_cols:
      temp[col] = temp[col].apply(lambda x: set_to_ten(x))
      analyses_after_censored[cutoff_time][file_name] = temp

# COMMAND ----------

for cutoff_time in analyses_after_censored:
    print(f"\t\t\t Checking {cutoff_time}-month cutoff analyses")
    for file_name in file_names:
        temp = analyses_after_censored[cutoff_time][file_name].copy()

        print(f"Checking {file_name}...")

        for col in count_cols:
            print(f"Minimum count in {col} is equal to or set to 10: {temp[col].min() >= 10}")
        
        print(f"Done with {file_name}\n")
    print(f"\t\t\t Done checking {cutoff_time}-month cutoff analyses\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert back to spark DataFrames and save files as csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving censored > 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

for cutoff_time in analyses_after_censored:
    for file_name in analyses_after_censored[cutoff_time]:
        temp = analyses_after_censored[cutoff_time][file_name].copy()
        temp.to_pickle(f"male_infertility_validation/revision_files/fx_{file_name}_{str(cutoff_time)}m_cutoff_c.pkl")

# COMMAND ----------

