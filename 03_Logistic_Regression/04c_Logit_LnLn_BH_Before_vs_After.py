# Databricks notebook source
# MAGIC %md
# MAGIC ## Create ln-ln plots for patients with male infertility vs vasectomy patients across logistic regression models
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This compares cohorts from the before analysis: < 6 months after diagnosis or procedure vs > 6 months after diagnosis or procedure
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 

# COMMAND ----------

# MAGIC %pip install boto3

# COMMAND ----------

!pip install UpSetPlot==0.6.0

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import pandas as pd
# MAGIC import seaborn as sns
# MAGIC import matplotlib.pyplot as plt
# MAGIC import numpy as np
# MAGIC import os
# MAGIC from scipy.stats import norm
# MAGIC from scipy.stats import chi2_contingency
# MAGIC from scipy.stats import spearmanr
# MAGIC from scipy.stats import pearsonr
# MAGIC import scipy.stats as stats
# MAGIC from math import log10, log2
# MAGIC #from tqdm import tqdm
# MAGIC import warnings
# MAGIC warnings.filterwarnings("ignore", category=FutureWarning) 
# MAGIC import upsetplot
# MAGIC # To merge mutliple dfs at once
# MAGIC import functools as ft
# MAGIC
# MAGIC import boto3
# MAGIC from io import BytesIO

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
# MAGIC ## Read in primary analysis logistic regression files (for phenotypes first diagnosed < 6 months after diagnosis/procedure and for phenotypes first diagnosed > 6 months after diagnosis/procedure)

# COMMAND ----------

before = pd.read_pickle("male_infertility_validation/tables/logit_results/before/primary.pkl")

after = pd.read_pickle("male_infertility_validation/tables/logit_results/after/primary.pkl")

# COMMAND ----------

# Make ln OR column
before['ln_OR_has_phenotype'] = np.log(before['odds_ratio_has_phenotype'])

after['ln_OR_has_phenotype'] = np.log(after['odds_ratio_has_phenotype'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## ln-ln plot

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make dataframe for comparing primary analyses (< 6 months after diagnosis/procedure vs > 6 months after diagnosis/procedure)

# COMMAND ----------

# Only include phenotypes that show up in both

phenotypes_both = set(before['phenotype']) & set(after['phenotype'])

before = before[before['phenotype'].isin(phenotypes_both)]
after = after[after['phenotype'].isin(phenotypes_both)]

# COMMAND ----------

before_vs_after = before[['phenotype', 
                          'ln_OR_has_phenotype', 
                          'significance_bh']].merge(after[['phenotype', 
                                                           'ln_OR_has_phenotype', 
                                                           'significance_bh']], 
                                                           on='phenotype', 
                                                           suffixes=('_before', '_after'))
before_vs_after.head(3)                  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine significance overlap between < 6 months vs > 6 months primary analyses

# COMMAND ----------

before_vs_after['sig_overlap'] = before_vs_after.apply(lambda x : sig_bh_overlap_det(df=x, primary='before', comp='after'), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate ln-ln plot
# MAGIC [Reference for changing figure size for seaborn plots](https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot)

# COMMAND ----------

before_vs_after['sig_overlap'].unique()

# COMMAND ----------

before_vs_after['sig_overlap'].value_counts()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(data=before_vs_after, x='ln_OR_has_phenotype_before', y='ln_OR_has_phenotype_after', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_before', 'mi_sig_after', 'mi_sig_both', 'con_sig_before', 'con_sig_after', 'con_sig_both'], palette=['whitesmoke', 'c', 'blue', 'midnightblue', 'lightcoral', 'red', 'firebrick'], alpha=0.4)

axes_min, axes_max = set_axes_bounds_ln(df_comp=before_vs_after,
                                     model_reference='before',
                                     model_compared='after')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=before_vs_after['ln_OR_has_phenotype_before'], b=before_vs_after['ln_OR_has_phenotype_after'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=before_vs_after['ln_OR_has_phenotype_before'], y=before_vs_after['ln_OR_has_phenotype_after'])
print(f"Pearson correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

# MAGIC %md
# MAGIC *Without `'Not Significant'` included*

# COMMAND ----------

# MAGIC %md
# MAGIC Adding annotation

# COMMAND ----------

before_vs_after_no_ns = before_vs_after[before_vs_after['sig_overlap'] != 'not_significant'].copy()

# COMMAND ----------

annotated_phe = list()

for overlap in before_vs_after_no_ns['sig_overlap'].unique():
  print(f"Phenotypes annotated for {overlap}")
  temp = before_vs_after_no_ns[before_vs_after_no_ns['sig_overlap'] == overlap]
  temp = temp.sort_values(by=['ln_OR_has_phenotype_before', 'ln_OR_has_phenotype_after']).head(2)
  for phenotype in temp['phenotype']:
    annotated_phe.append(phenotype)
  display(temp)

# COMMAND ----------

before_vs_after_no_ns['annotate'] = before_vs_after_no_ns['phenotype'].apply(lambda x: 1 if x in list(annotated_phe) else 0)

# COMMAND ----------

annotate = True

# COMMAND ----------

before_vs_after_no_ns['sig_overlap'] = before_vs_after_no_ns['sig_overlap'].replace({'mi_sig_before' : 'mi - before only',
                                                                                     'mi_sig_after' : 'mi - after only',
                                                                                     'mi_sig_both' : 'mi - both',
                                                                                     'con_sig_before' : 'con - before only',
                                                                                     'con_sig_after' : 'con - after only',
                                                                                     'con_sig_both' : 'con - both'})
                                
fig, ax = plt.subplots(figsize=(12,12))
  
ax = sns.scatterplot(data=before_vs_after_no_ns, 
                     x='ln_OR_has_phenotype_before', 
                     y='ln_OR_has_phenotype_after', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - before only',
                                'mi - after only',
                                'mi - both',     
                                'con - before only', 
                                'con - after only', 
                                'con - both'], 
                     palette=['pink',
                              'hotpink', 
                              'purple', 
                              'powderblue',  
                              'dodgerblue',   
                              'midnightblue'], 
                     alpha=0.6)

texts = []
kwargs = dict(color='k', fontweight='medium')

# Add annotation (20220111)
# source: https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas-dataframe/39374693
if annotate:
  for idx, row in before_vs_after_no_ns.iterrows():
    if row['annotate'] == 1:
        texts.append(ax.annotate(row['phenotype'], (row['ln_OR_has_phenotype_before'], row['ln_OR_has_phenotype_after']), fontsize=18, **kwargs))
        
if len(texts) > 0:
  adjust_text(texts, arrowprops=dict(arrowstyle="-", 
                                      color='k'))

axes_min, axes_max = set_axes_bounds_ln(df_comp=before_vs_after,
                                     model_reference='before',
                                     model_compared='after')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'ln(Odds Ratio)''\n< 6 months', fontsize=18)
ax.set_ylabel('> 6 months\n'r'ln(Odds Ratio)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_title('ln-ln plot\n< 6 months vs > 6 months\nafter diagnosis or procedure', fontsize=20, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)
plt.axhline(y=0, color='k', lw=0.5, linestyle='dotted')
plt.axvline(x=0, color='k', lw=0.5, linestyle='dotted')

plt.legend(loc=(0.01,0.782), fontsize=16)

if save:
  print('Saving ln-ln plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_b4_vs_aft_ln.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

