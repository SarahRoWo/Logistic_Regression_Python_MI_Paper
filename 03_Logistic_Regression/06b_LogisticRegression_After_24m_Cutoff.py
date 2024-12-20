# Databricks notebook source
# MAGIC %md
# MAGIC ## Compare diagnoses for patients with male infertility vs patients who do not have male infertility diagnosis
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC ### Only includes patients with at least two years of follow-up time
# MAGIC ### Only includes diagnoses from within two years of follow-up time
# MAGIC

# COMMAND ----------

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

cutoff_time = 24

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
# MAGIC ## Read in demographics files

# COMMAND ----------

mi_demo = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")

vas_only_demo = pd.read_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent diagnoses files with corresponding phecodes

# COMMAND ----------

# Patients with male infertility
mi_phe = pd.read_pickle("male_infertility_validation/diagnoses/mi_phe_w_diag_deltas.pkl")

# Vasectomy only patients
vas_only_phe = pd.read_pickle("male_infertility_validation/diagnoses/vas_phe_w_diag_deltas.pkl")

# COMMAND ----------

mi_demo_cutoff_after, mi_phe_cutoff_after = filter_by_cutoff(demo=mi_demo, phe=mi_phe, cutoff_time=cutoff_time, phe_analysis_cutoff='after')

# COMMAND ----------

mi_phe_cutoff_after.shape

# COMMAND ----------

vas_only_demo_cutoff_after, vas_only_phe_cutoff_after = filter_by_cutoff(demo=vas_only_demo, phe=vas_only_phe, cutoff_time=cutoff_time, phe_analysis_cutoff='after')

# COMMAND ----------

vas_only_phe_cutoff_after.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain combined diagnoses and demographics

# COMMAND ----------

diag_combined = combine_diagnoses(df_case=mi_phe_cutoff_after, df_con=vas_only_phe_cutoff_after)
demo_combined = combine_demographics(mi_demo=mi_demo_cutoff_after, con_demo=vas_only_demo_cutoff_after)

# COMMAND ----------

# Combined demographics equals the number of male infertility + vasectomy patients after cutoff
demo_combined['person_id'].nunique() == (mi_demo_cutoff_after['person_id'].nunique() + vas_only_demo_cutoff_after['person_id'].nunique())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Only keep phenotypes that overlap for male infertility and control patients

# COMMAND ----------

phenotypes_both = set(mi_phe_cutoff_after['phenotype'].unique()) & set(vas_only_phe_cutoff_after['phenotype'].unique())

# COMMAND ----------

# remove None if in phentoypes_both
None in phenotypes_both

# COMMAND ----------

diag_combined = diag_combined[diag_combined['phenotype'].isin(phenotypes_both)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run logistic regression analyses

# COMMAND ----------

# MAGIC %md
# MAGIC ### crude

# COMMAND ----------

crude_results = run_logistic_regresssion(diag_combined=diag_combined, demo_combined=demo_combined, formula='has_mi ~ has_phenotype')

# COMMAND ----------

# MAGIC %md
# MAGIC ### primary

# COMMAND ----------

primary_results = run_logistic_regresssion(diag_combined=diag_combined, demo_combined=demo_combined, formula='has_mi ~ has_phenotype + mi_or_vas_est_age')

# COMMAND ----------

# MAGIC %md
# MAGIC ### sensitivity analysis - social determinants of health

# COMMAND ----------

sdoh_results = run_logistic_regresssion(diag_combined=diag_combined, demo_combined=demo_combined, formula='has_mi ~ has_phenotype + mi_or_vas_est_age + race + ethnicity')

# COMMAND ----------

# MAGIC %md
# MAGIC ### sensitivity analysis - hospital utilization

# COMMAND ----------

hosp_results = run_logistic_regresssion(diag_combined=diag_combined, demo_combined=demo_combined, formula='has_mi ~ has_phenotype + mi_or_vas_est_age + num_visits_after + emr_months_after')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get patient counts for each phenotype

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter for phenotypes obtained after male infertility diagnosis or vasectomy procedure

# COMMAND ----------

total_mi = mi_demo_cutoff_after['person_id'].nunique()
total_mi

# COMMAND ----------

total_vas_only = vas_only_demo_cutoff_after['person_id'].nunique()
total_vas_only

# COMMAND ----------

# 20241028 modification note below:
# Filtered for phenotypes represented in logistic regression models
mi_phe_aft_pre = mi_phe_cutoff_after[mi_phe_cutoff_after['phe_time_after'] == 1].copy()
mi_phe_aft = mi_phe_aft_pre[mi_phe_aft_pre['phenotype'].isin(phenotypes_both)].copy()

vas_only_phe_aft_pre = vas_only_phe_cutoff_after[vas_only_phe_cutoff_after['phe_time_after'] == 1].copy()
vas_only_phe_aft = vas_only_phe_aft_pre[vas_only_phe_aft_pre['phenotype'].isin(phenotypes_both)].copy()

# COMMAND ----------

mi_diag_count = countPtsDiagnosis_Dict_LR(mi_phe_aft, total_mi, ['phenotype'])
vas_only_diag_count = countPtsDiagnosis_Dict(vas_only_phe_aft, total_vas_only, ['phenotype'])

# COMMAND ----------

alldiagcount = dict()

for n in diagkeys:
      alldiagcount[n] = mi_diag_count[n].merge(vas_only_diag_count[n], how='outer', on=n, suffixes=('_mi','_con'))
      alldiagcount[n] = alldiagcount[n].set_index(n)
      nanreplace = dict(zip(list(alldiagcount[n].columns) , [0,total_mi,0,total_vas_only]))
      alldiagcount[n] =  alldiagcount[n].fillna(value=nanreplace)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean dataframes, add patient counts for each phenotype, and add significance

# COMMAND ----------

results_dict = dict()

all_results = [crude_results, primary_results, sdoh_results, hosp_results]
names = ['crude', 'primary', 'sdoh', 'hosp']

for result, name in zip(all_results, names):
  print(f"Cleaning {name} analysis...")
  temp = result.copy()
  print('Cleaning column names...')
  temp = clean_col_names(temp)
  print('Making log10_OR_has_phenotype column...')
  temp['log10_OR_has_phenotype'] = np.log10(temp['odds_ratio_has_phenotype'])
  print('Making log10_pval_has_phenotype column...')
  temp['-log10_pval_has_phenotype'] = -1*(np.log10(temp['pval_has_phenotype'])) 
  print('Adding patient counts...')
  temp = temp.merge(alldiagcount[n], on='phenotype')
  print('Adding columns for BH correction')
  temp['pval_BH_adj_sig'], temp['pval_BH_adj_has_phenotype'] = fdrcorrection(pvals=list(temp['pval_has_phenotype']))
  print('Adding bonferroni-corrected significance...')
  temp = significance_bc(temp)
  print('Adding Benjamini-Hochberg significance...')
  temp['significance_bh'] = temp.apply(significance_bh, axis=1)
  results_dict[name] = temp
  print('Done.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add phecode_category associated with each phentoype

# COMMAND ----------

# MAGIC %md
# MAGIC ### 'Import' phecode notebooks

# COMMAND ----------

phecodes = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

# COMMAND ----------

results_final = dict()

for result in results_dict:
  print(f"Adding phecode category for {result}...")
  temp = results_dict[result].copy()
  temp = temp.merge(phecodes[['phenotype', 'excl_phenotypes']], on='phenotype', how='left').copy().drop_duplicates().reset_index(drop=True)
  print('Changing excl_phenotypes column name to phecode_category...')
  temp = temp.rename({'excl_phenotypes' : 'phecode_category'}, axis=1)
  print("Replacing None phecode_category values to 'null'...")
  temp['phecode_category'] = temp['phecode_category'].fillna(value='null')
  results_final[result] = temp
  print('Done.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore value counts for each analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonferroni-corrected

# COMMAND ----------

for analysis in results_final:
  print(f"Overview of results for {analysis} analysis:")
  temp = results_final[analysis]
  display(temp['significance'].value_counts())
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benajamini-Hochberg 

# COMMAND ----------

for analysis in results_final:
  print(f"Overview of results for {analysis} analysis:")
  temp = results_final[analysis]
  display(temp['significance_bh'].value_counts())
  print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save

# COMMAND ----------

for analysis in results_final:
  print(f"Saving results for {analysis} analysis:")
  temp = results_final[analysis]
  temp.to_pickle(f"male_infertility_validation/revision_files/fx_{analysis}_{str(cutoff_time)}m_cutoff.pkl")
  print('Done.\n')

# COMMAND ----------

