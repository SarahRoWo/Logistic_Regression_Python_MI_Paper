# Databricks notebook source
# MAGIC %md
# MAGIC ## Save censored logistic regression analyses 
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This includes the following logistic regression analyses
# MAGIC
# MAGIC - < 6 months after diagnosis/procedure:
# MAGIC 1. `has_mi ~ has_phenotype` 
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before` 
# MAGIC
# MAGIC - > 6 months after diagnosis/procedure:
# MAGIC 1. `has_mi ~ has_phenotype` 
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_after + months_in_EMR_after` 

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import pandas as pd
# MAGIC import seaborn as sns
# MAGIC import matplotlib.pyplot as plt
# MAGIC import numpy as np
# MAGIC import os
# MAGIC from scipy.stats import norm
# MAGIC from scipy.stats import chi2_contingency
# MAGIC import scipy.stats as stats
# MAGIC from math import log10, log2
# MAGIC #from tqdm import tqdm
# MAGIC import warnings
# MAGIC warnings.filterwarnings("ignore", category=FutureWarning) 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import chi2_contingency
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import log10, log2
from tqdm import tqdm
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

# MAGIC %run MI_Functions.py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in logistic regression analysis files

# COMMAND ----------

# MAGIC %md
# MAGIC ### < 6 months after diagnosis/procedure

# COMMAND ----------

analyses_before_pd = dict()

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

for file_name in file_names:
  analyses_before_pd[file_name] = pd.read_pickle(f"male_infertility_validation/tables/logit_results/before/{file_name}.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### > 6 months after diagnosis/procedure

# COMMAND ----------

analyses_after_pd = dict()

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

for file_name in file_names:
  analyses_after_pd[file_name] = pd.read_pickle(f"male_infertility_validation/tables/logit_results/after/{file_name}.pkl")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Censor low counts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Columns that will be censored

# COMMAND ----------

count_cols = analyses_before_pd['primary'].filter(regex='Count')
count_cols.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Censoring < 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

analyses_before_pd_censored = dict()

for analysis in analyses_before_pd:
  temp = analyses_before_pd[analysis].copy()

  for col in count_cols:
    temp[col] = temp[col].apply(lambda x: set_to_ten(x))
    analyses_before_pd_censored[analysis] = temp

# COMMAND ----------

# Check
for analysis in analyses_before_pd_censored:
  temp = analyses_before_pd_censored[analysis].copy()
  
  print(f"Checking {analysis}...")

  for col in count_cols:
    print(f"Minimum count in {col} is equal to or set to 10: {temp[col].min() >= 10}")
  
  print('Done.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Censoring > 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

analyses_after_pd_censored = dict()

for analysis in analyses_after_pd:
  temp = analyses_after_pd[analysis].copy()

  for col in count_cols:
    temp[col] = temp[col].apply(lambda x: set_to_ten(x))
    analyses_after_pd_censored[analysis] = temp

# COMMAND ----------

# Check
for analysis in analyses_after_pd_censored:
  temp = analyses_after_pd_censored[analysis].copy()
  
  print(f"Checking {analysis}...")

  for col in count_cols:
    print(f"Minimum count in {col} is equal to or set to 10: {temp[col].min() >= 10}")
  
  print('Done.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert back to spark DataFrames and save files as csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving censored < 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

for analysis in analyses_before_pd_censored:

  temp = analyses_before_pd_censored[analysis].copy()
  
  print(f"Saving {analysis} as csv file...")

  temp.to_csv(f"male_infertility_validation/tables/logit_results/before_censored/{analysis}.csv")
  print('Saved.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving censored > 6 months after diagnosis/procedure logistic regression analyses

# COMMAND ----------

for analysis in analyses_after_pd_censored:

  temp = analyses_after_pd_censored[analysis].copy()

  print(f"Saving {analysis} as csv file...")

  temp.to_csv(f"male_infertility_validation/tables/logit_results/after_censored/{analysis}.csv")
  print('Saved.\n')

# COMMAND ----------

