# Databricks notebook source
# MAGIC %md
# MAGIC ## Create log-log and upset plots for patients with male infertility vs vasectomy patients across logistic regression models
# MAGIC ### Includes phenotypes from ICD9CM and ICD10CM diagnoses
# MAGIC
# MAGIC ### This compares cohorts from the following logistic regression models 
# MAGIC 1. `has_mi ~ has_phenotype` 
# MAGIC 2. `has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before` 

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
# MAGIC ## Read in pertinent logistic regression files

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyses' file name descriptions
# MAGIC 1. `'crude' : has_mi ~ has_phenotype` 
# MAGIC 2. `'primary' : has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `'sdoh' : has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `'hosp' : has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before`

# COMMAND ----------

analyses_pd = dict()

# COMMAND ----------

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

for file_name in file_names:
  analyses_pd[file_name] = pd.read_pickle(f"male_infertility_validation/tables/logit_results/before/{file_name}.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log-log plots

# COMMAND ----------

for analysis in analyses_pd:
   temp = analyses_pd[analysis]
  
   # make log10_OR_has_phenotype column for analyses
   temp['log10_OR_has_phenotype'] = np.log10(temp['odds_ratio_has_phenotype'])

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
                                           'log10_OR_has_phenotype', 
                                           'significance_bh']].merge(analyses_pd['crude'][['phenotype',               
                                                                                           'log10_OR_has_phenotype', 'significance_bh']], 
                                                                                           on='phenotype', 
                                                                                           suffixes=('_primary', '_crude'))
primary_vs_crude.head(3)                  

# COMMAND ----------

# MAGIC %md
# MAGIC 2. primary vs sdoh

# COMMAND ----------

primary_vs_sdoh = analyses_pd['primary'][['phenotype', 
                                          'log10_OR_has_phenotype', 
                                          'significance_bh']].merge(analyses_pd['sdoh'][['phenotype', 
                                                                                         'log10_OR_has_phenotype', 'significance_bh']], 
                                                                                         on='phenotype', 
                                                                                         suffixes=('_primary', '_sdoh'))
primary_vs_sdoh.head(3)                  

# COMMAND ----------

# MAGIC %md
# MAGIC 3. primary vs hosp

# COMMAND ----------

primary_vs_hosp = analyses_pd['primary'][['phenotype', 
                                          'log10_OR_has_phenotype', 
                                          'significance_bh']].merge(analyses_pd['hosp'][['phenotype', 
                                                                                         'log10_OR_has_phenotype', 'significance_bh']], 
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
# MAGIC ### Generate log-log plots
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

ax = sns.scatterplot(data=primary_vs_crude, x='log10_OR_has_phenotype_primary', y='log10_OR_has_phenotype_crude', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_crude', 'mi_sig_both', 'con_sig_crude', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'powderblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_crude,
                                     model_reference='primary',
                                     model_compared='crude')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_crude['log10_OR_has_phenotype_primary'], b=primary_vs_crude['log10_OR_has_phenotype_crude'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_crude['log10_OR_has_phenotype_primary'], y=primary_vs_crude['log10_OR_has_phenotype_crude'])
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
                     x='log10_OR_has_phenotype_primary', 
                     y='log10_OR_has_phenotype_crude', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - primary only',
                                'mi - crude only', 
                                'mi - both',      
                                'con - crude only', 
                                'con - both'], 
                     palette=['hotpink', 
                              'pink', 
                              'purple', 
                              'powderblue',     
                              'midnightblue'], 
                     alpha=0.6)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_crude,
                                     model_reference='primary',
                                     model_compared='crude')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('crude analysis\n'r'$\log_{10}$(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('log-log plot\nprimary vs crude analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(0.02,0.73), fontsize=14)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_vs_crude_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
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

ax = sns.scatterplot(data=primary_vs_sdoh, x='log10_OR_has_phenotype_primary', y='log10_OR_has_phenotype_sdoh', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_sdoh', 'mi_sig_both', 'con_sig_primary', 'con_sig_sdoh', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'dodgerblue', 'powderblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_sdoh,
                                     model_reference='primary',
                                     model_compared='sdoh')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_sdoh['log10_OR_has_phenotype_primary'], b=primary_vs_sdoh['log10_OR_has_phenotype_sdoh'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_sdoh['log10_OR_has_phenotype_primary'], y=primary_vs_sdoh['log10_OR_has_phenotype_sdoh'])
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
                     x='log10_OR_has_phenotype_primary', 
                     y='log10_OR_has_phenotype_sdoh', 
                     s=25, 
                     hue='sig_overlap', 
                     hue_order=['mi - primary only',
                                'mi - SDoH only', 
                                'mi - both',     
                                'con - primary only', 
                                'con - SDoH only', 
                                'con - both'], 
                     palette=['hotpink', 
                              'pink', 
                              'purple', 
                              'dodgerblue',     
                              'powderblue',
                              'midnightblue'], 
                     alpha=0.6)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_sdoh,
                                     model_reference='primary',
                                     model_compared='sdoh')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('SDoH sensitivity analysis\n'r'$\log_{10}$(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('log-log plot\nprimary vs SDoH sensitivity analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(0.02,0.68), fontsize=14)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_vs_sdoh_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')
  
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Version of plot above for Figure 1

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,8))
  
ax = sns.scatterplot(data=primary_vs_sdoh_no_ns, 
                     x='log10_OR_has_phenotype_primary', 
                     y='log10_OR_has_phenotype_sdoh', 
                     s=150, 
                     hue='sig_overlap', 
                     hue_order=['mi - primary only',
                                'mi - SDoH only', 
                                'mi - both',   
                                'con - primary only',   
                                'con - SDoH only', 
                                'con - both'], 
                     palette=['hotpink', 
                              'pink', 
                              'purple', 
                              'dodgerblue',     
                              'powderblue',
                              'midnightblue'], 
                     alpha=0.6,
                     legend=False)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_sdoh,
                                     model_reference='primary',
                                     model_compared='sdoh')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('SDoH sensitivity analysis\n'r'$\log_{10}$(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_vs_sdoh_b4_fig1.pdf", format='pdf', dpi=300, bbox_inches='tight')
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

ax = sns.scatterplot(data=primary_vs_hosp, x='log10_OR_has_phenotype_primary', y='log10_OR_has_phenotype_hosp', s=20, hue='sig_overlap', hue_order=['not_significant', 'mi_sig_primary', 'mi_sig_hosp', 'mi_sig_both', 'con_sig_primary', 'con_sig_both'], palette=['whitesmoke', 'hotpink', 'pink', 'purple', 'dodgerblue', 'midnightblue'], alpha=0.4)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_hosp,
                                     model_reference='primary',
                                     model_compared='hosp')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(1.05,0.775))
plt.show()

# COMMAND ----------

statistic, p_value = spearmanr(a=primary_vs_hosp['log10_OR_has_phenotype_primary'], b=primary_vs_hosp['log10_OR_has_phenotype_hosp'])
print(f"Spearman correlation coefficient statistic is {statistic} \np-value is {p_value}")

# COMMAND ----------

statistic, p_value = pearsonr(x=primary_vs_hosp['log10_OR_has_phenotype_primary'], y=primary_vs_hosp['log10_OR_has_phenotype_hosp'])
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
                     x='log10_OR_has_phenotype_primary', 
                     y='log10_OR_has_phenotype_hosp', 
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

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_hosp,
                                     model_reference='primary',
                                     model_compared='hosp')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('hospital utilization sensitivity analysis\n'r'$\log_{10}$(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('log-log plot\nprimary vs hospital utilization sensitivity analysis\n', fontsize=18, fontweight='bold')

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

plt.legend(loc=(0.02,0.727), fontsize=14)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_vs_hosp_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

primary_vs_hosp_no_ns = primary_vs_hosp[primary_vs_hosp['sig_overlap'] != 'not_significant'].copy()

primary_vs_hosp_no_ns['sig_overlap'] = primary_vs_hosp_no_ns['sig_overlap'].replace({'mi_sig_hosp' : 'mi - hosp. util. only',
                                                                                     'mi_sig_primary' : 'mi - primary only',
                                                                                     'mi_sig_both' : 'mi - both',
                                                                                     'con_sig_primary' : 'con - primary only',
                                                                                     'con_sig_both' : 'con - both'})
                                
fig, ax = plt.subplots(figsize=(10,8))
  
ax = sns.scatterplot(data=primary_vs_hosp_no_ns, 
                     x='log10_OR_has_phenotype_primary', 
                     y='log10_OR_has_phenotype_hosp', 
                     s=150, 
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
                     alpha=0.6,
                     legend=False)

axes_min, axes_max = set_axes_bounds(df_comp=primary_vs_hosp,
                                     model_reference='primary',
                                     model_compared='hosp')

ax.set_xlim([axes_min, axes_max])
ax.set_ylim([axes_min, axes_max])
ax.set_xlabel(r'$\log_{10}$(Odds Ratio)''\nprimary analysis', fontsize=16)
ax.set_ylabel('hospital utilization sensitivity analysis\n'r'$\log_{10}$(Odds Ratio)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.set_aspect('equal', adjustable='box')

plt.plot([axes_min, axes_max], [axes_min, axes_max], color='k', lw=0.5)

if save:
  print('Saving log-log plot...')
  plt.savefig("male_infertility_validation/figures/logit/loglog/fx_vs_hosp_b4_fig1.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upset plots

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create dataframe containing overlapping phenotypes and their significance across matching strategies

# COMMAND ----------

primary_sig = analyses_pd['primary'][['phenotype', 'phecode_category', 'significance_bh']].copy()
primary_sig = primary_sig.rename({'significance_bh' : 'sig_primary'}, axis=1)
primary_sig = primary_sig.set_index('phenotype')
primary_sig = primary_sig.sort_values(by='phenotype')

sdoh_sig = analyses_pd['sdoh'][['phenotype', 'phecode_category', 'significance_bh']].copy()
sdoh_sig = sdoh_sig.rename({'significance_bh' : 'sig_sdoh'}, axis=1)
sdoh_sig = sdoh_sig.set_index('phenotype')
sdoh_sig = sdoh_sig.sort_values(by='phenotype')

hosp_sig = analyses_pd['hosp'][['phenotype', 'phecode_category', 'significance_bh']].copy()
hosp_sig = hosp_sig.rename({'significance_bh' : 'sig_hosp'}, axis=1)
hosp_sig = hosp_sig.set_index('phenotype')
hosp_sig = hosp_sig.sort_values(by='phenotype')
                                                                                
all_sig = primary_sig.join([sdoh_sig['sig_sdoh'], hosp_sig['sig_hosp']], how='inner')
all_sig['phecode_category'] = all_sig['phecode_category'].fillna('None')
display(all_sig.head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determine overall significance of the phenotypes between matching strategies for patients with male infertility and controls

# COMMAND ----------

all_sig['Significance_MI'] = all_sig.apply(lambda x: overall_significance_mi(x), axis=1)
all_sig['Significance_Con'] = all_sig.apply(lambda x: overall_significance_con(x), axis=1)

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

upset = upsetplot.from_memberships(sig_overlaps_MI_list[1:],
                                   data=sig_values_MI[1:])

upsetplot.plot(upset, show_counts=True)

# https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines-in-matplotlib-python
plt.grid(False)

plt.ylabel('Significant Phenotype \n Intersections', fontweight='bold')
plt.title('Number of Significant Phenotypes \n Across Logit Models \n \n Patients with Male Infertility \n', fontweight='bold')

if save:
  print('Saving Upset plot...')
  plt.savefig("male_infertility_validation/figures/logit/upset/fx_upset_mi_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

upset = upsetplot.from_memberships(sig_overlaps_MI_list[1:],
                                   data=sig_values_MI[1:])

upsetplot.plot(upset)

# https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines-in-matplotlib-python
plt.grid(False)

plt.ylabel('Significant Phenotype \n Intersections', fontweight='bold')
plt.title('Number of Significant Phenotypes \n Across Logit Models \n \n Patients with Male Infertility \n', fontweight='bold')

if save:
  print('Saving Upset plot...')
  plt.savefig("male_infertility_validation/figures/logit/upset/fx_upset_mi_b4_2.pdf", format='pdf', dpi=300, bbox_inches='tight')
  print('Saved.')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upset plot for control patients

# COMMAND ----------

upset = upsetplot.from_memberships(sig_overlaps_Con_list[1:],
                                   data=sig_values_Con[1:])

upsetplot.plot(upset, show_counts=True)

# https://stackoverflow.com/questions/45148704/how-to-hide-axes-and-gridlines-in-matplotlib-python
plt.grid(False)

plt.ylabel('Significant Phenotype \n Intersections', fontweight='bold')
plt.title('Number of Significant Phenotypes \n Across Logit Models \n \n Control Patients \n', fontweight='bold')

if save:
  print('Saving Upset plot...')
  plt.savefig("/dbfs/FileStore/fx_upset_con_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
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
      print(f"Saving mi_before_{overlap_folder_name}")
      temp.to_csv(f"male_infertility_validation/tables/upset/mi_before_{overlap_folder_name}.csv")
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
      print(f"Saving con_before_{overlap_folder_name}")
      temp.to_csv(f"male_infertility_validation/tables/upset/con_before_{overlap_folder_name}.csv")
      print('Saved.\n')

# COMMAND ----------

