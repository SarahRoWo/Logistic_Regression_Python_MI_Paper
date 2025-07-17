# Databricks notebook source
# MAGIC %md
# MAGIC ## Create ln-ln plots for patients with male infertility vs vasectomy patients across logistic regression models
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This compares cohorts from the following logistic regression models 
# MAGIC 1. `has_mi ~ has_phenotype` 
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before` 

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib_venn as mv
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import scipy.stats as stats
from math import log10, log2
#from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
# To merge mutliple dfs at once
import functools as ft

# COMMAND ----------

pd.set_option('display.max_rows', 50)
np.set_printoptions(threshold=50)

# COMMAND ----------

diagkeys = ['phenotype']

# COMMAND ----------

# Whether to save figures and spark DataFrames for paper
save = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent logistic regression files

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyses' file name descriptions
# MAGIC 1. `'crude' : has_mi ~ has_phenotype` 
# MAGIC 2. `'primary' : has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `'sdoh' : has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `'hosp' : has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before`

# COMMAND ----------

analyses = dict()

# COMMAND ----------

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

for file_name in file_names:
  analyses_pd[file_name] = pd.read_pickle(f"male_infertility_validation/tables/logit_results/before/{file_name}.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ln-ln plots

# COMMAND ----------

analyses_pd = dict()

for analysis in analyses:
   temp = analyses[analysis].toPandas()
  
   # make ln_OR_has_phenotype column for analyses
   temp['ln_OR_has_phenotype'] = np.log(temp['odds_ratio_has_phenotype'])

   analyses_pd[analysis] = temp

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make dataframes for the following comparisons:
# MAGIC 1. primary vs crude
# MAGIC 2. primary vs sdoh
# MAGIC 3. primary vs hosp

# COMMAND ----------

# MAGIC %md
# MAGIC 1. primary vs crude

# COMMAND ----------

primary_vs_crude = analyses_pd['primary'][['phenotype', 
                                           'ln_OR_has_phenotype', 
                                           'significance_bh']].merge(analyses_pd['crude'][['phenotype',               
                                                                                           'ln_OR_has_phenotype', 'significance_bh']], 
                                                                                           on='phenotype', 
                                                                                           suffixes=('_primary', '_crude'))
primary_vs_crude.head(3)                  

# COMMAND ----------

# MAGIC %md
# MAGIC 2. primary vs sdoh

# COMMAND ----------

primary_vs_sdoh = analyses_pd['primary'][['phenotype', 
                                          'ln_OR_has_phenotype', 
                                          'significance_bh']].merge(analyses_pd['sdoh'][['phenotype', 
                                                                                         'ln_OR_has_phenotype', 'significance_bh']], 
                                                                                         on='phenotype', 
                                                                                         suffixes=('_primary', '_sdoh'))
primary_vs_sdoh.head(3)                  

# COMMAND ----------

# MAGIC %md
# MAGIC 3. primary vs hosp

# COMMAND ----------

primary_vs_hosp = analyses_pd['primary'][['phenotype', 
                                          'ln_OR_has_phenotype', 
                                          'significance_bh']].merge(analyses_pd['hosp'][['phenotype', 
                                                                                         'ln_OR_has_phenotype', 'significance_bh']], 
                                                                                         on='phenotype', 
                                                                                         suffixes=('_primary', '_hosp'))
primary_vs_hosp.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine significance overlap between primary analysis and analysis it is being compared to

# COMMAND ----------

primary_vs_crude['sig_overlap'] = primary_vs_crude.apply(lambda x : sig_bh_overlap_det(df=x, comp='crude'), axis=1)
primary_vs_sdoh['sig_overlap'] = primary_vs_sdoh.apply(lambda x : sig_bh_overlap_det(df=x, comp='sdoh'), axis=1)
primary_vs_hosp['sig_overlap'] = primary_vs_hosp.apply(lambda x : sig_bh_overlap_det(df=x, comp='hosp'), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate ln-ln plots
# MAGIC [Reference for changing figure size for seaborn plots](https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. primary vs crude

# COMMAND ----------

primary_vs_crude['sig_overlap'].unique()

# COMMAND ----------

primary_vs_crude['sig_overlap'].value_counts()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(data=primary_vs_crude, x='ln_OR_has_phenotype_primary', y='ln_OR_has_phenotype_crude', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_crude', 'mi_sig_both', 'con_sig_crude', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'powderblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_crude,
                                     model_reference='primary',
                                     model_compared='crude')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_crude['ln_OR_has_phenotype_primary'], b=primary_vs_crude['ln_OR_has_phenotype_crude'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_crude['ln_OR_has_phenotype_primary'], y=primary_vs_crude['ln_OR_has_phenotype_crude'])
print(f"Pearson correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC *Without `'Not Significant'` included*

# COMMAND ----------

primary_vs_crude_no_ns = primary_vs_crude[primary_vs_crude['sig_overlap'] != 'not_significant'].copy()

primary_vs_crude_no_ns['sig_overlap'] = primary_vs_crude_no_ns['sig_overlap'].replace({'mi_sig_primary' : 'mi - primary only',
                                                                                       'mi_sig_crude' : 'mi - crude only',
                                                                                       'mi_sig_both' : 'mi - both',
                                                                                       'con_sig_crude' : 'con - crude only',
                                                                                       'con_sig_both' : 'con - both'})
                                
fig, ax = plt.subplots(figsize=(10,8))
  
ax = sns.scatterplot(data=primary_vs_crude_no_ns, 
                     x='ln_OR_has_phenotype_primary', 
                     y='ln_OR_has_phenotype_crude', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - crude only',
                                'mi - primary only',
                                'mi - both',      
                                'con - crude only', 
                                'con - both'], 
                     palette=['pink',
                              'hotpink', 
                              'purple', 
                              'powderblue',     
                              'midnightblue'], 
                     alpha=0.6)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_crude,
                                     model_reference='primary',
                                     model_compared='crude')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'ln(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('crude analysis\n'r'ln(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('ln-ln plot\nprimary vs crude analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)
plt.axhline(y=0, color='k', lw=0.5, linestyle='dotted')
plt.axvline(x=0, color='k', lw=0.5, linestyle='dotted')


plt.legend(loc=(0.02,0.735), fontsize=14)

if save:
  print('Saving ln-ln plot...')
  plt.savefig("male_infertility_validation/revision_files/fx_vs_crude_b4_ln.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')
  
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. primary vs sdoh

# COMMAND ----------

primary_vs_sdoh['sig_overlap'].unique()

# COMMAND ----------

primary_vs_sdoh['sig_overlap'].value_counts()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(data=primary_vs_sdoh, x='ln_OR_has_phenotype_primary', y='ln_OR_has_phenotype_sdoh', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_sdoh', 'mi_sig_both', 'con_sig_primary', 'con_sig_sdoh', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'dodgerblue', 'powderblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_sdoh,
                                     model_reference='primary',
                                     model_compared='sdoh')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_sdoh['ln_OR_has_phenotype_primary'], b=primary_vs_sdoh['ln_OR_has_phenotype_sdoh'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_sdoh['ln_OR_has_phenotype_primary'], y=primary_vs_sdoh['ln_OR_has_phenotype_sdoh'])
print(f"Pearson correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC *Without `'Not Significant'` included*

# COMMAND ----------

primary_vs_sdoh_no_ns = primary_vs_sdoh[primary_vs_sdoh['sig_overlap'] != 'not_significant'].copy()

primary_vs_sdoh_no_ns['sig_overlap'] = primary_vs_sdoh_no_ns['sig_overlap'].replace({'mi_sig_sdoh' : 'mi - SDoH only',
                                                                                     'mi_sig_primary' : 'mi - primary only',
                                                                                     'mi_sig_both' : 'mi - both',
                                                                                     'con_sig_sdoh' : 'con - SDoH only',
                                                                                     'con_sig_primary' : 'con - primary only',
                                                                                     'con_sig_both' : 'con - both'})
                                
fig, ax = plt.subplots(figsize=(10,8))
  
ax = sns.scatterplot(data=primary_vs_sdoh_no_ns, 
                     x='ln_OR_has_phenotype_primary', 
                     y='ln_OR_has_phenotype_sdoh', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - SDoH only',
                                'mi - primary only',
                                'mi - both',   
                                'con - SDoH only',  
                                'con - primary only', 
                                'con - both'], 
                     palette=['pink',
                              'hotpink',  
                              'purple',   
                              'powderblue',
                              'dodgerblue', 
                              'midnightblue'], 
                     alpha=0.6)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_sdoh,
                                     model_reference='primary',
                                     model_compared='sdoh')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'ln(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('SDoH sensitivity analysis\n'r'ln(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('ln-ln plot\nprimary vs SDoH sensitivity analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)
plt.axhline(y=0, color='k', lw=0.5, linestyle='dotted')
plt.axvline(x=0, color='k', lw=0.5, linestyle='dotted')

plt.legend(loc=(0.02,0.69), fontsize=14)

if save:
  print('Saving ln-ln plot...')
  plt.savefig("male_infertility_validation/revision_files/fx_vs_sdoh_b4_ln.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')
  
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 3. primary vs hosp

# COMMAND ----------

primary_vs_hosp['sig_overlap'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC *With `'Not Significant'` included*

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(data=primary_vs_hosp, x='ln_OR_has_phenotype_primary', y='ln_OR_has_phenotype_hosp', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_hosp', 'mi_sig_both', 'con_sig_primary', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'dodgerblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_hosp,
                                     model_reference='primary',
                                     model_compared='hosp')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_hosp['ln_OR_has_phenotype_primary'], b=primary_vs_hosp['ln_OR_has_phenotype_hosp'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_hosp['ln_OR_has_phenotype_primary'], y=primary_vs_hosp['ln_OR_has_phenotype_hosp'])
print(f"Pearson correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC *Without `'Not Significant'` included*

# COMMAND ----------

primary_vs_hosp_no_ns = primary_vs_hosp[primary_vs_hosp['sig_overlap'] != 'not_significant'].copy()

primary_vs_hosp_no_ns['sig_overlap'] = primary_vs_hosp_no_ns['sig_overlap'].replace({'mi_sig_hosp' : 'mi - hosp. util. only',
                                                                                     'mi_sig_primary' : 'mi - primary only',
                                                                                     'mi_sig_both' : 'mi - both',
                                                                                     'con_sig_primary' : 'con - primary only',
                                                                                     'con_sig_both' : 'con - both'})
                                
fig, ax = plt.subplots(figsize=(10,8))
  
ax = sns.scatterplot(data=primary_vs_hosp_no_ns, 
                     x='ln_OR_has_phenotype_primary', 
                     y='ln_OR_has_phenotype_hosp', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - hosp. util. only', 
                                'mi - primary only',
                                'mi - both',     
                                'con - primary only',
                                'con - both'], 
                     palette=['pink', 
                              'hotpink', 
                              'purple',     
                              'dodgerblue',
                              'midnightblue'], 
                     alpha=0.6)

axes_min, axes_max = set_axes_bounds_ln(df_comp=primary_vs_hosp,
                                     model_reference='primary',
                                     model_compared='hosp')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'ln(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('hospital utilization sensitivity analysis\n'r'ln(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('ln-ln plot\nprimary vs hospital utilization sensitivity analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)
plt.axhline(y=0, color='k', lw=0.5, linestyle='dotted')
plt.axvline(x=0, color='k', lw=0.5, linestyle='dotted')

plt.legend(loc=(0.02,0.737), fontsize=14)

if save:
  print('Saving ln-ln plot...')
  plt.savefig("male_infertility_validation/revision_files/fx_vs_hosp_b4_ln.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

