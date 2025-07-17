# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## UMAP visualization of patients' diagnoses based on ethnicity categorization
# MAGIC
# MAGIC Note: This visualizes all phenotypes not related to infertility

# COMMAND ----------

import os
os.chdir('/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI')
from MI_Functions import *

# COMMAND ----------

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
feature = 'ethnicity'

# COMMAND ----------

all_demo_df = pd.read_pickle("male_infertility_validation/tables/umap/demo_all.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in 'X', the 2D representations of patients' diagnoses

# COMMAND ----------

X = pd.read_pickle("male_infertility_validation/tables/umap/mi_vas_only.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create X_embedded (numpy array)

# COMMAND ----------

X_embedded = make_X_embedded(X)

# COMMAND ----------

display(X.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in 'y', which preserves each patient's male infertility status (it is not preserved after performing dimensionality reduction)

# COMMAND ----------

y_all = pd.read_pickle("male_infertility_validation/tables/umap/y_all.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert y (which can be any feature) to series of feature of interest

# COMMAND ----------

y = y_all[feature]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize UMAP

# COMMAND ----------

visualize_UMAP_data(X_embedded=X_embedded, 
                    y=y,
                    hue_order=['Hispanic or Latino', 'Not Hispanic or Latino', 'No matching concept'], 
                    feature='Ethnicity Categorization',
                    bbox_to_anchor=(0.393, 0.15),
                    alpha=0.15,
                    figure_size=(10, 10),
                    label_axes=False,
                    palette=['navy', 'darkturquoise', 'deeppink'],
                    save=True,
                    file_name='fx_umap_eth')

# COMMAND ----------

make_UMAP_violin_plots(X_embedded=X_embedded, 
                       y_values=y.values, 
                       order=['Hispanic or Latino', 'Not Hispanic or Latino', 'No matching concept'],
                       palette=['navy', 'darkturquoise', 'deeppink'],
                       save=True,
                       filename_UMAP_1='fx_umap_eth_violin1',
                       filename_UMAP_2='fx_umap_eth_violin2')

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

# First component
statistic_1, pvalue_1 = mstats.kruskalwallis(np.asarray(val1['axis_1']), 
                                              np.asarray(val2['axis_1']),
                                              np.asarray(val3['axis_1']))

print(f"Kruskal-Wallis Result for First Component: \nStatistic is: {statistic_1} \np-value is: {pvalue_1}")
if pvalue_1 < 0.05:
  print('Significant')
else:
  print('Not Significant')
print('\n')

# Second component
statistic_2, pvalue_2 = mstats.kruskalwallis(np.asarray(val1['axis_2']), 
                                             np.asarray(val2['axis_2']),
                                             np.asarray(val3['axis_2']))

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
                    val3[['axis_1', feature]]],
                    axis=0)

axis_2 = pd.concat([val1[['axis_2', feature]],
                    val2[['axis_2', feature]],
                    val3[['axis_2', feature]]],
                    axis=0)

if pvalue_1 < 0.05:
  # First axis post-hoc
  print("Posthoc Dunn's test: axis 1")
  display(posthoc_dunn(a=axis_1, val_col='axis_1', group_col=feature, p_adjust='bonferroni').reset_index())
  if save_dunn:
    print("Saving Dunn's test for axis 1...")
    dunn = posthoc_dunn(a=axis_1, val_col='axis_1', group_col=feature, p_adjust='bonferroni').reset_index()
    dunn.to_csv("male_infertility_validation/tables/umap/dunns_test/dunn_ethnicity_axis1.csv")
    print('Saved.\n')
if pvalue_2 < 0.05:
  # Second axis post-hoc
  print("Posthoc Dunn's test: axis 2")
  display(posthoc_dunn(a=axis_2, val_col='axis_2', group_col=feature, p_adjust='bonferroni').reset_index())
  if save_dunn:
    print("Saving Dunn's test for axis 2...")
    dunn = posthoc_dunn(a=axis_2, val_col='axis_2', group_col=feature, p_adjust='bonferroni').reset_index()
    #dunn_spark = spark.createDataFrame(dunn)
    dunn.to_csv("male_infertility_validation/tables/umap/dunns_test/dunn_ethnicity_axis2.csv")
    print('Saved.')

# COMMAND ----------



