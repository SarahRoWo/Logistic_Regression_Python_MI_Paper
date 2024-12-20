# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Notebook for Step 1: Identifying outlier and nonoutlier patients

# COMMAND ----------

import pandas as pd
import numpy as np
import os

# COMMAND ----------

if not os.path.isdir("male_infertility_validation/revision_files"):
    os.mkdir("male_infertility_validation/revision_files")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in 'X', the 2D representations of patients' diagnoses

# COMMAND ----------

#X_spark = spark.read.format("parquet").load("male_infertility_validation/tables/umap/mi_vas_only_before").orderBy('index')
X = pd.read_pickle("male_infertility_validation/tables/umap/mi_vas_only_before.pkl").sort_values(by='index').copy()

# COMMAND ----------

display(X.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify x axis coordinates that will be used to filter outlier patients

# COMMAND ----------

his_plt = X[1].plot.hist()

# COMMAND ----------

# obtain counts and bins for the histogram
counts_bins = np.histogram(X[0])

# COMMAND ----------

# use the count and bin values to get number of patients in the outlier cluster and x-axis range, respectively

counts_bins_df = pd.DataFrame(data=counts_bins).transpose().rename({0 : "count", 1 : "bin"}, axis=1).copy()
counts_bins_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter X for outlier and nonoutlier patients

# COMMAND ----------

# filter for outlier patients
X_outlier_pre = X.copy()

X_outlier = X_outlier_pre[ (X_outlier_pre[0] > 4)]

print(X_outlier.shape)

# COMMAND ----------

# filter for nonoutlier patients

X_nonoutlier = X[~X["index"].isin(X_outlier["index"])].copy()
print(X_nonoutlier.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in 'y', which preserves each patient's male infertility status (it is not preserved after performing dimensionality reduction) as well as demographic features

# COMMAND ----------

#y_spark = spark.read.format("parquet").load("male_infertility_validation/tables/umap/y_all_before").orderBy('index')
y_all = pd.read_pickle("male_infertility_validation/tables/umap/y_all_before.pkl").sort_values(by='index').copy()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Merge outlier and nonoutlier patients with ```y_all```

# COMMAND ----------

# Merge X_outlier and y_all

Xy_outlier = X_outlier.merge(y_all, on='index').copy()

print(Xy_outlier.shape)

# COMMAND ----------

# Merge X_nonoutlier and y_all

Xy_nonoutlier = X_nonoutlier.merge(y_all, on='index').copy()

print(Xy_nonoutlier.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get person_ids for outliers and nonoutliers, stratified by male infertility status

# COMMAND ----------

# Get person ids for patients with male infertility and vasectomy for outliers
Xy_outlier_mi = Xy_outlier[Xy_outlier['male infertility status'] == 'male infertility patient'].copy()
print(f"Number of outlier male infertility patients: {Xy_outlier_mi.shape[0]}")

Xy_outlier_vas = Xy_outlier[Xy_outlier['male infertility status'] == 'control (vasectomy patient)'].copy()
print(f"Number of outlier vasectomy patients: {Xy_outlier_vas.shape[0]}")

# COMMAND ----------

# Get person ids for patients with male infertility and vasectomy for nonoutliers
Xy_nonoutlier_mi = Xy_nonoutlier[Xy_nonoutlier['male infertility status'] == 'male infertility patient'].copy()
print(f"Number of nonoutlier male infertility patients: {Xy_nonoutlier_mi.shape[0]}")

Xy_nonoutlier_vas = Xy_nonoutlier[Xy_nonoutlier['male infertility status'] == 'control (vasectomy patient)'].copy()
print(f"Number of nonoutlier vasectomy patients: {Xy_nonoutlier_vas.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save

# COMMAND ----------

# outliers

Xy_outlier.to_pickle("male_infertility_validation/revision_files/Xy_outlier.pkl")
Xy_outlier_mi.to_pickle("male_infertility_validation/revision_files/Xy_outlier_mi.pkl")
Xy_outlier_vas.to_pickle("male_infertility_validation/revision_files/Xy_outlier_vas.pkl")


#spark.createDataFrame(Xy_outlier).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_outlier")

#spark.createDataFrame(Xy_outlier_mi).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_outlier_mi")

#spark.createDataFrame(Xy_outlier_vas).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_outlier_vas")

# COMMAND ----------

# nonoutliers

Xy_nonoutlier.to_pickle("male_infertility_validation/revision_files/Xy_nonoutlier.pkl")
Xy_nonoutlier_mi.to_pickle("male_infertility_validation/revision_files/Xy_nonoutlier_mi.pkl")
Xy_nonoutlier_vas.to_pickle("male_infertility_validation/revision_files/Xy_nonoutlier_vas.pkl")

#spark.createDataFrame(Xy_nonoutlier).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_nonoutlier")

#spark.createDataFrame(Xy_nonoutlier_mi).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_nonoutlier_mi")

#spark.createDataFrame(Xy_nonoutlier_vas).write.mode("overwrite").parquet("male_infertility_validation/revision_files/Xy_nonoutlier_vas")

# COMMAND ----------


