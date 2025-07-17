# Databricks notebook source
# MAGIC %md
# MAGIC ##Explore logistic regression models for patients with male infertility vs patients who have undergone vasectomy-related procedure
# MAGIC
# MAGIC ## 1. Create volcano plots for patients with male infertility vs patients who do not have male infertility diagnosis
# MAGIC ## 2. Create manhattan plots showing p-value of phenotypes per category
# MAGIC
# MAGIC ### Includes phenotypes mapped from ICD9CM and ICD10CM diagnoses

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
# MAGIC from math import log10, log10
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
# MAGIC ## Read in pertinent analysis files

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyses' file name descriptions
# MAGIC 1. `'crude' : has_mi ~ has_phenotype` 
# MAGIC 2. `'primary' : has_mi ~ has_phenotype + estimated_age + location_source_value` 
# MAGIC 3. `'sdoh' : has_mi ~ has_phenotype + estimated_age + location_source_value + race + ethnicity + ADI` 
# MAGIC 4. `'hosp' : has_mi ~ has_phenotype + estimated_age + location_source_value + num_visits_before + months_in_EMR_before`

# COMMAND ----------

analyses_pd = dict()

file_names = ['crude', 
              'primary',
              'sdoh', 
              'hosp']

for file_name in file_names:
  analyses_pd[file_name] = pd.read_pickle(f"male_infertility_validation/tables/logit_results/before/{file_name}.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Annotate top ORs for volcano plots (both vasectomy and male infertility)

# COMMAND ----------

for analysis in analyses_pd:
  temp = analyses_pd[analysis]

  temp2 = temp[temp['significance_bh'] != 'not_significant']
  
  mi_annotate = temp2.sort_values(by='log10_OR_has_phenotype', ascending=False).head(10)
  con_annotate = temp2.sort_values(by='log10_OR_has_phenotype').head(5)
  all_annotate = pd.concat([mi_annotate['phenotype'], con_annotate['phenotype']]).reset_index(drop=True).to_frame()
  
  temp['annotate'] = temp['phenotype'].apply(lambda x: 1 if x in list(all_annotate['phenotype']) else 0)

  analyses_pd[analysis] = temp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Replace values that are infinite in `-log10_pval_has_phenotype` column and 0 in `pval_has_phenotype` column

# COMMAND ----------

# MAGIC %md
# MAGIC ### `-log10_pval_has_phenotype` is the only column with infinite values

# COMMAND ----------

df_copy = analyses_pd['primary'].copy()

# Filter for numeric columns only
# Ref: https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
numeric_df = df_copy.select_dtypes(include=np.number)

# Find columns with infinite values
cols = numeric_df.columns.to_series()[np.isinf(numeric_df).any()]
print(f"Columns with infinite values: {cols[0]}")

# COMMAND ----------

analyses_visualize_volcano = dict()

for analysis in analyses_pd:
  temp = analyses_pd[analysis].copy()
  
  if cols.shape[0] > 0:
    print(color.BOLD + f"Replacing infinity value with largest finite value in -log10_pval_has_phenotype column for {analysis} analysis..." + color.END)

    # Obtain largest non infinite value in -log10_pval_has_phenotype column
    log10_column = temp['-log10_pval_has_phenotype'].copy()
    log10_array = np.asarray(log10_column)
    log10_array_finite = log10_array[np.isfinite(log10_array)]
    log10_max_value = np.nanmax(log10_array_finite)
    print(f"Largest finite -log10_pval_has_phenotype value that will replace infinity value: {log10_max_value}")

    # Obtain smallest pval_has_phenotype greater than 0
    pval_column = temp[temp['pval_has_phenotype'] > 0]['pval_has_phenotype']
    pval_min_value = pval_column.min()
    print(f"Smallest non-zero pval_has_phenotype value that will replace 0 value: {pval_min_value}")

    # Replace infinity value with log10_max_value in -log10_pval_has_phenotype column
    temp['-log10_pval_has_phenotype'] = temp['-log10_pval_has_phenotype'].replace({np.inf : log10_max_value})
    print(f"Checking largest value in -log10_pval_has_phenotype column is now finite: {temp['-log10_pval_has_phenotype'].max()}")

    # Replace 0 value with pval_min_value in pval_has_phenotype column
    temp['pval_has_phenotype'] = temp['pval_has_phenotype'].replace({0 : pval_min_value})
    print(f"Checking smallest value in pval_has_phenotype column is now non-zero: {temp['pval_has_phenotype'].min()}")

    # Save new DataFrame
    print(f"Saving {analysis} in analyses_visualize_volcano...")
    analyses_visualize_volcano[analysis] = temp
  else:
    # Save new DataFrame
    print(f"Saving {analysis} in analyses_visualize_volcano...")
    analyses_visualize_volcano[analysis] = temp
  
  print('Done.\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volcano Plot for primary analysis

# COMMAND ----------

volcano(analyses_pd['primary'], 
        'primary', s=20, 
        annotate=True, 
        annotatefontsize=22, 
        dims=(15,10), 
        legend_loc=(0.3235, -0.30),
        save=True,
        file_name_save='male_infertility_validation/figures/logit/volcano/fx_vol_b4')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manhattan plot - primary analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Annotate the 10 phenotypes with lowest p-values for patients with male infertility

# COMMAND ----------

analyses_visualize_manhattan = dict()

for analysis in analyses_visualize_volcano:
  # identify the top 15 phenotypes associated with male infertility
  temp = analyses_visualize_volcano['primary'].copy()
  temp = temp[(temp['significance_bh'] == 'mi_significant') | (temp['significance_bh'] == 'male infertility significant')]

  # top 10 phenotypes with lowest p-value
  mi_annotate_mhat = temp.sort_values(by='-log10_pval_has_phenotype', ascending=False).head(10)['phenotype']

  # updated annotate column for primary analysis will be for p-values
  temp2 = analyses_visualize_volcano['primary'].copy()
  temp2['annotate'] = temp2['phenotype'].apply(lambda x: 1 if x in list(mi_annotate_mhat) else 0)

  analyses_visualize_manhattan[analysis] = temp2

# COMMAND ----------

# MAGIC %md
# MAGIC #### Primary analysis annotation

# COMMAND ----------

analyses_visualize_manhattan['primary'][analyses_visualize_manhattan['primary']['annotate'] == 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sort values by `-log10_pval_has_phenotype` to find appropriate line breaks for plot

# COMMAND ----------

display(analyses_visualize_manhattan['primary'].sort_values(by='-log10_pval_has_phenotype', ascending=False))

# COMMAND ----------

# Find p-value cutoff
cutoff = analyses_visualize_manhattan['primary'][analyses_visualize_manhattan['primary']['pval_BH_adj_has_phenotype'] <= 0.05]
cutoff_pval = cutoff['pval_has_phenotype'].max()
print(cutoff_pval)

# COMMAND ----------

analyses_visualize_manhattan.to_pickle


fig = mhat_two_line_breaks(df=analyses_visualize_manhattan['primary'], 
                           logp='-log10_pval_has_phenotype',
                           groupby_col='phecode_category', 
                           #dim=(40, 10), 
                           dim=(20,10),
                           rows=9,
                           columns=2,
                           nrowstop=1, # number of rows for top subplot
                           nrowsmid=2, # number of rows for middle subplot\
                           topmin=230, # min y-axis value for top subplot
                           topmax=240, # max y-axis value for top subplot
                           midmin=100, # min y-axis value for middle subplot
                           midmax=120, # max y-axis value for middle subplot
                           botmin=0, # min y-axis value for bottom subplot
                           botmax=40, # max y-axis value for bottom subplot 
                           sig_pval_line=True,
                           sig_pval=cutoff_pval,
                           axlabelfontsize=31, 
                           axtickfontsize=28, 
                           ar=90,
                           dotsize=25, 
                           figtitle=(f"Male Infertility vs. Vasectomy\nManhattan Plot\n"),
                           annotate=True,
                           annotatefontsize=25, #22.25
                           autoalign=False,
                           expand_text=(0.4, 0.5),
                           expand_points=(0.5, 0.5),
                           save=False)  
print('Saving outside function...')
fig.savefig("male_infertility_validation/figures/logit/manhattan/fx_mhat_b4.pdf", format='pdf', dpi=300, bbox_inches='tight')
print('Saved.')                    

# COMMAND ----------

