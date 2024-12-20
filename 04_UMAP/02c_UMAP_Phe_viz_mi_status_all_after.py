# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## UMAP visualization of patients' diagnoses based on male infertility status
# MAGIC
# MAGIC Note: This visualizes all phenotypes not related to infertility that were first diagnosed > 6 months after diagnosis/procedure

# COMMAND ----------


# COMMAND ----------
runfile('/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI/MI_Functions.py', wdir='/Users/fengxie/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI')

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
feature = 'male infertility status'

# COMMAND ----------

all_demo_df = pd.read_pickle("male_infertility_validation/tables/umap/demo_all_after.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in 'X', the 2D representations of patients' diagnoses

# COMMAND ----------

X = pd.read_pickle("male_infertility_validation/tables/umap/mi_vas_only_after.pkl").sort_values(by='index').copy()

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

y_all = pd.read_pickle("male_infertility_validation/tables/umap/y_all_after.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert y (which can be any feature) to series of feature of interest

# COMMAND ----------

y = y_all[feature]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize UMAP

# COMMAND ----------

visualize_UMAP_data(X_embedded=X_embedded, 
                    y=y,
                    hue_order=y.unique(), 
                    feature=feature.title(),
                    bbox_to_anchor=(0.535, 1.0),
                    alpha=0.135,
                    palette=['hotpink', 'dodgerblue'],
                    figure_size=(10, 10),
                    label_axes=False,
                    save=True,
                    file_name='fx_umap_mi_aft')

# COMMAND ----------

make_UMAP_violin_plots(X_embedded=X_embedded, 
                       y_values=y.values, 
                       palette=['hotpink', 'dodgerblue'],
                       save=True,
                       filename_UMAP_1='fx_umap_mi_aft_violin1',
                       filename_UMAP_2='fx_umap_mi_aft_violin2')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mann-Whitney U test

# COMMAND ----------

print(color.BOLD + f"Mann-Whitney U test for logistic regression cohort:" + color.END)
cat_1 = X_embedded[y.values == y.unique()[0] ,:]
cat_2 = X_embedded[y.values == y.unique()[1] ,:]
print('Axis 1: ', mannwhitneyu(cat_1[:,0], cat_2[:,0]))
print('Axis 2: ', mannwhitneyu(cat_1[:,1], cat_2[:,1]))
print('\n')

# COMMAND ----------

