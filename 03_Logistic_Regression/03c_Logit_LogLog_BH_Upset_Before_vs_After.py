# Databricks notebook source
# MAGIC %md
# MAGIC ## Create log-log and upset plots for patients with male infertility vs vasectomy patients across logistic regression models
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This compares cohorts from the before analysis: < 6 months after diagnosis or procedure vs > 6 months after diagnosis or procedure
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 

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

# MAGIC %run MI_Functions.py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in primary analysis logistic regression files (for phenotypes first diagnosed < 6 months after diagnosis/procedure and for phenotypes first diagnosed > 6 months after diagnosis/procedure)

# COMMAND ----------

before = pd.read_pickle("male_infertility_validation/tables/logit_results/before/primary.pkl")

after = pd.read_pickle("male_infertility_validation/tables/logit_results/after/primary.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log-log plot

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
                          'log10_OR_has_phenotype', 
                          'significance_bh']].merge(after[['phenotype', 
                                                           'log10_OR_has_phenotype', 
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
# MAGIC ### Generate log-log plot
# MAGIC [Reference for changing figure size for seaborn plots](https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot)

# COMMAND ----------

before_vs_after['sig_overlap'].unique()

# COMMAND ----------

before_vs_after['sig_overlap'].value_counts()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))

ax = sns.scatterplot(data=before_vs_after, x='log10_OR_has_phenotype_before', y='log10_OR_has_phenotype_after', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_before', 'mi_sig_after', 'mi_sig_both', 'con_sig_before', 'con_sig_after', 'con_sig_both'], palette=['whitesmoke', 'c', 'blue', 'midnightblue', 'lightcoral', 'red', 'firebrick'], alpha=0.4)

axes_min, axes_max = set_axes_bounds(df_comp=before_vs_after,
                                     model_reference='before',
                                     model_compared='after')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=before_vs_after['log10_OR_has_phenotype_before'], b=before_vs_after['log10_OR_has_phenotype_after'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=before_vs_after['log10_OR_has_phenotype_before'], y=before_vs_after['log10_OR_has_phenotype_after'])
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
  temp = temp.sort_values(by=['log10_OR_has_phenotype_before', 'log10_OR_has_phenotype_after']).head(2)
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
                     x='log10_OR_has_phenotype_before', 
                     y='log10_OR_has_phenotype_after', 
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
        texts.append(ax.annotate(row['phenotype'], (row['log10_OR_has_phenotype_before'], row['log10_OR_has_phenotype_after']), fontsize=18, **kwargs))
        
if len(texts) > 0:
  adjust_text(texts, arrowprops=dict(arrowstyle="-", 
                                      color='k'))

axes_min, axes_max = set_axes_bounds(df_comp=before_vs_after,
                                     model_reference='before',
                                     model_compared='after')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\n< 6 months', fontsize=18)
ax.set_ylabel('> 6 months\n'r'$\log_{10}$(Odds Ratio)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_title('log-log plot\n< 6 months vs > 6 months\nafter diagnosis or procedure', fontsize=20, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(0.01,0.77), fontsize=16)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_b4_vs_aft.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upset plots

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create dataframe containing overlapping phenotypes and their significance across matching strategies

# COMMAND ----------

before_sig = before[['phenotype', 'phecode_category', 'significance_bh']].copy()
before_sig = before_sig.rename({'significance_bh' : 'sig_before'}, axis=1)
before_sig = before_sig.set_index('phenotype')

after_sig = after[['phenotype', 'phecode_category', 'significance_bh']].copy()
after_sig = after_sig.rename({'significance_bh' : 'sig_after'}, axis=1)
after_sig = after_sig.set_index('phenotype')
                                                                                  
all_sig = before_sig.join([after_sig['sig_after']], how='inner')
all_sig['phecode_category'] = all_sig['phecode_category'].fillna('None')
display(all_sig.head())

# COMMAND ----------

all_sig['Significance_MI'] = all_sig.apply(lambda x: overall_significance_mi(x, suffixes=['_before', '_after']), axis=1)
all_sig['Significance_Con'] = all_sig.apply(lambda x: overall_significance_con(x, suffixes=['_before', '_after']), axis=1)

# COMMAND ----------

all_sig['Sig_Upset_MI'] = all_sig['Significance_MI'].apply(abv)
all_sig['Sig_Upset_Con'] = all_sig['Significance_Con'].apply(abv)

# COMMAND ----------

all_sig['Sig_Upset_MI'].value_counts().sort_index()

# COMMAND ----------

all_sig['Sig_Upset_Con'].value_counts().sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare results for upset plots

# COMMAND ----------

# MAGIC %md
# MAGIC Male Infertility

# COMMAND ----------

sig_overlaps_MI = all_sig['Sig_Upset_MI'].value_counts().sort_index()
sig_values_MI = list(sig_overlaps_MI)

sig_overlaps_MI_list = list(sig_overlaps_MI.index)
sig_overlaps_MI_list

# COMMAND ----------

# MAGIC %md
# MAGIC Control patients

# COMMAND ----------

sig_overlaps_Con = all_sig['Sig_Upset_Con'].value_counts().sort_index()
sig_values_Con = list(sig_overlaps_Con)

sig_overlaps_Con_list = list(sig_overlaps_Con.index)
sig_overlaps_Con_list

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upset plot for patients with male infertility

# COMMAND ----------

save = True
upset = upsetplot.from_memberships(sig_overlaps_MI_list[1:],
                                   data=sig_values_MI[1:])

upsetplot.plot(upset, show_counts=True)

# https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines-in-matplotlib-python
plt.grid(False)

plt.ylabel('Significant Phenotype \n Intersections', fontweight='bold')
plt.title('Number of Significant Phenotypes \n Across Logit Models \n \n Patients with Male Infertility \n', fontweight='bold')

if save:
  print('Saving Upset plot...')
  plt.savefig("male_infertility_validation/figures/logit/upset/fx_upset_mi_b4_aft.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upset plot for control patients

# COMMAND ----------

save = True
upset = upsetplot.from_memberships(sig_overlaps_Con_list[1:],
                                   data=sig_values_Con[1:])

upsetplot.plot(upset, show_counts=True)

# https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines-in-matplotlib-python
plt.grid(False)

plt.ylabel('Significant Phenotype \n Intersections', fontweight='bold')
plt.title('Number of Significant Phenotypes \n Across Logit Models \n \n Control Patients \n', fontweight='bold')

if save:
  print('Saving Upset plot...')
  plt.savefig("male_infertility_validation/figures/logit/upset/fx_upset_con_b4_aft.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overlapping phenotypes for patients with male infertility

# COMMAND ----------

for sig_overlap in all_sig['Significance_MI'].unique():
  if sig_overlap != ():
    print(f"Phenotypes in common with {sig_overlap}:")
    temp = all_sig[all_sig['Significance_MI'] == sig_overlap].reset_index()[['phenotype', 'phecode_category']]
    display(temp['phecode_category'].value_counts())
    display(temp)
    if save:
      # Generate file name if saving
      overlap_names = list()
      for overlap in sig_overlap:
        overlap_names.append(overlap[16:])
      overlap_folder_name = '_'.join(overlap_names)
      print(f"Saving mi_before_vs_after_{overlap_folder_name}")
      temp.to_csv(f"male_infertility_validation/tables/upset/mi_before_vs_after_{overlap_folder_name}.csv")
      print('Saved.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overlapping phenotypes for control patients

# COMMAND ----------

for sig_overlap in all_sig['Significance_Con'].unique():
  if sig_overlap != ():
    print(f"Phenotypes in common with {sig_overlap}:")
    temp = all_sig[all_sig['Significance_Con'] == sig_overlap].reset_index()[['phenotype', 'phecode_category']]
    display(temp['phecode_category'].value_counts())
    display(temp)
    if save:
      # Generate file name if saving
      overlap_names = list()
      for overlap in sig_overlap:
        overlap_names.append(overlap[16:])
      overlap_folder_name = '_'.join(overlap_names)
      print(f"Saving con_before_vs_after_{overlap_folder_name}")
      temp.to_csv(f"male_infertility_validation/tables/upset/con_before_vs_after_{overlap_folder_name}.csv")
      print('Saved.\n')

# COMMAND ----------

