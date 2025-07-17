# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## UMAP visualization of patients' diagnoses based on age
# MAGIC
# MAGIC Note: This visualizes all phenotypes not related to infertility

# COMMAND ----------

runfile('/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI/MI_Functions.py', wdir='/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI')

# COMMAND ----------

# Restart state after running above for the first time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.core.multiarray
import numpy as np
import os
import re
import scipy
from scipy import stats
from scipy.stats import mstats
from scipy.stats import mannwhitneyu
from scikit_posthocs import posthoc_dunn
import matplotlib
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run MI_Functions.py

# COMMAND ----------

# feature corresponds to column of interest 
feature = 'age bracket'

# COMMAND ----------

all_demo_df = pd.read_pickle("male_infertility_validation/tables/umap/demo_all.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in 'X', the 2D representations of patients' diagnoses

# COMMAND ----------

X = pd.read_pickle("male_infertility_validation/tables/umap/mi_vas_only.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create X_embedded (numpy array)

# COMMAND ----------

X_embedded = make_X_embedded(X)

# COMMAND ----------

display(X.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in 'y', which preserves each patient's male infertility status (it is not preserved after performing dimensionality reduction) as well as demographic features

# COMMAND ----------

y_all = pd.read_pickle("male_infertility_validation/tables/umap/y_all.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC  %md
# MAGIC ### Convert y (which can be any feature) to series of feature of interest

# COMMAND ----------

# MAGIC %md
# MAGIC #### First, create age_bracket function

# COMMAND ----------

def age_bracket(age_series):
  """
  Parameters
  __________
  age_series : pandas Series
    Contains ages of patients

  Returns
  _______
  age_bracket : pandas Series
    Contains age bracket of patients
  """

  if age_series < 30:
    age_bracket = '< 30'
  elif (age_series >= 30) and (age_series < 40):
    age_bracket = '30 - 39'
  elif (age_series >= 40) and (age_series < 50):
    age_bracket = '40 - 49'
  elif (age_series >= 50) and (age_series < 60):
    age_bracket = '50 - 59'
  elif (age_series >= 60) and (age_series < 70):
    age_bracket = '60 - 69'
  elif (age_series >= 70) and (age_series < 80):
    age_bracket = '70 - 79'
  else:
    if age_series >= 80:
      age_bracket = '>= 80'
    else:
      print('Check age')
  
  return age_bracket

# COMMAND ----------

# Get quintiles for number of visits
y_all['age bracket'] = y_all['estimated_age'].apply(age_bracket)

y = y_all[feature]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize UMAP

# COMMAND ----------

visualize_UMAP_data(X_embedded=X_embedded, 
                    y=y,
                    hue_order=['< 30', 
                               '30 - 39',
                               '40 - 49', 
                               '50 - 59',
                               '60 - 69',  
                               '70 - 79',
                               '>= 80'], 
                    feature=feature.title(),
                    bbox_to_anchor=(0.201, 0.3),
                    alpha=0.15,
                    figure_size=(10, 10),
                    label_axes=False,
                    palette='Dark2',
                    save=True,
                    file_name='fx_umap_age')

# COMMAND ----------

make_UMAP_violin_plots(X_embedded=X_embedded, 
                       y_values=y.values, 
                       order=['< 30', 
                              '30 - 39',
                              '40 - 49', 
                              '50 - 59',
                              '60 - 69',  
                              '70 - 79',
                              '>= 80'],
                       palette='Dark2',
                       save=True,
                       filename_UMAP_1='fx_umap_age_violin1',
                       filename_UMAP_2='fx_umap_age_violin2')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kruskal-Wallis, followed by Dunn's test

# COMMAND ----------

save_dunn = True

# COMMAND ----------

print(color.BOLD + f"Kruskal-Wallis test" + color.END)
y_categories = y.unique()
  
print(f"Categories compared: {y.unique()}")
print('\n')
  
# Create arrays for each category
val1 = X_embedded[y.values == y_categories[0], :]
val1 = convert_UMAP_array_to_df(val1)
val1[feature] = y_categories[0]
  
val2 = X_embedded[y.values == y_categories[1], :]
val2 = convert_UMAP_array_to_df(val2)
val2[feature] = y_categories[1]
  
val3 = X_embedded[y.values == y_categories[2], :]
val3 = convert_UMAP_array_to_df(val3)
val3[feature] = y_categories[2]
  
val4 = X_embedded[y.values == y_categories[3], :]
val4 = convert_UMAP_array_to_df(val4)
val4[feature] = y_categories[3]
  
val5 = X_embedded[y.values == y_categories[4], :]
val5 = convert_UMAP_array_to_df(val5)
val5[feature] = y_categories[4]

val6 = X_embedded[y.values == y_categories[4], :]
val6 = convert_UMAP_array_to_df(val5)
val6[feature] = y_categories[5]

val7 = X_embedded[y.values == y_categories[4], :]
val7 = convert_UMAP_array_to_df(val5)
val7[feature] = y_categories[6]

# First component
statistic_1, pvalue_1 = mstats.kruskalwallis(np.asarray(val1['axis_1']), 
                                             np.asarray(val2['axis_1']), 
                                             np.asarray(val3['axis_1']), 
                                             np.asarray(val4['axis_1']),
                                             np.asarray(val5['axis_1']),
                                             np.asarray(val6['axis_1']),
                                             np.asarray(val7['axis_1']))

print(f"Kruskal-Wallis Result for First Component: \nStatistic is: {statistic_1} \np-value is: {pvalue_1}")
if pvalue_1 < 0.05:
  print('Significant')
else:
  print('Not Significant')
print('\n')

# Second component
statistic_2, pvalue_2 = mstats.kruskalwallis(np.asarray(val1['axis_2']), 
                                             np.asarray(val2['axis_2']),
                                             np.asarray(val3['axis_2']),
                                             np.asarray(val4['axis_2']),
                                             np.asarray(val5['axis_2']),
                                             np.asarray(val6['axis_2']),
                                             np.asarray(val7['axis_2']))
                                              
print(f"Kruskal-Wallis Result for Second Component: \nStatistic is: {statistic_2} \np-value is: {pvalue_2}")
if pvalue_2 < 0.05:
  print('Significant')
else:
  print('Not Significant')
print('\n')
  
# Posthoc Dunn's test
# Make two dataframes: one for axis 1, other for axis 2 
axis_1 = pd.concat([val1[['axis_1', feature]],
                    val2[['axis_1', feature]],
                    val3[['axis_1', feature]],
                    val4[['axis_1', feature]],
                    val5[['axis_1', feature]],
                    val6[['axis_1', feature]],
                    val7[['axis_1', feature]]],
                    axis=0)

axis_2 = pd.concat([val1[['axis_2', feature]],
                    val2[['axis_2', feature]],
                    val3[['axis_2', feature]],
                    val4[['axis_2', feature]],
                    val5[['axis_2', feature]],
                    val6[['axis_2', feature]],
                    val7[['axis_2', feature]]],
                    axis=0)
  
if pvalue_1 < 0.05:
  # First axis post-hoc
  print("Posthoc Dunn's test: axis 1")
  display(posthoc_dunn(a=axis_1, val_col='axis_1', group_col=feature, p_adjust='bonferroni').reset_index())
  if save_dunn:
    print("Saving Dunn's test for axis 1...")
    dunn = posthoc_dunn(a=axis_1, val_col='axis_1', group_col=feature, p_adjust='bonferroni').reset_index()
    dunn.to_csv("male_infertility_validation/tables/umap/dunns_test/dunn_age_axis1.csv")
    print('Saved.\n')
if pvalue_2 < 0.05:
  # Second axis post-hoc
  print("Posthoc Dunn's test: axis 2")
  display(posthoc_dunn(a=axis_2, val_col='axis_2', group_col=feature, p_adjust='bonferroni').reset_index())
  if save_dunn:
    print("Saving Dunn's test for axis 2...")
    dunn = posthoc_dunn(a=axis_2, val_col='axis_2', group_col=feature, p_adjust='bonferroni').reset_index()
    dunn.to_csv("male_infertility_validation/tables/umap/dunns_test/dunn_age_axis2.csv")
    print('Saved.')

# COMMAND ----------

