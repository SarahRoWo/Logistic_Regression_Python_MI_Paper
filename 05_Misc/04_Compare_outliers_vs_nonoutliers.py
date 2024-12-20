# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook for Step 4: Obtain number of phecodes per patient for outliers and nonoutliers and run Mann-Whitney U tests to compare

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in phecode files

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outliers

# COMMAND ----------

# unstratified patients
#phe_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/phe_outlier")
#phe_outlier = phe_outlier_spark.toPandas()

phe_outlier = pd.read_pickle("male_infertility_validation/revision_files/phe_outlier.pkl")

# male infertility patients
#mi_phe_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/mi_phe_outlier")
#mi_phe_outlier = mi_phe_outlier_spark.toPandas()

mi_phe_outlier = pd.read_pickle("male_infertility_validation/revision_files/mi_phe_outlier.pkl")

# vasectomy patients
#vas_phe_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/vas_phe_outlier")
#vas_phe_outlier = vas_phe_outlier_spark.toPandas()

vas_phe_outlier = pd.read_pickle("male_infertility_validation/revision_files/vas_phe_outlier.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Nonoutliers

# COMMAND ----------

# unstratified patients
#phe_nonoutlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/phe_nonoutlier")
#phe_nonoutlier = phe_nonoutlier_spark.toPandas()

phe_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/phe_nonoutlier")

# male infertility patients
#mi_phe_nonoutlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/mi_phe_nonoutlier")
#mi_phe_nonoutlier = mi_phe_nonoutlier_spark.toPandas()

mi_phe_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/mi_phe_nonoutlier")

# vasectomy patients
#vas_phe_nonoutlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/vas_phe_nonoutlier")
#vas_phe_nonoutlier = vas_phe_nonoutlier_spark.toPandas()

vas_phe_nonoutlier = pd.read_pickle("male_infertility_validation/revision_files/vas_phe_nonoutlier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain phecodes before and after 6-month cutoff for unstratified, male infertility, and vasectomy patients

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outliers

# COMMAND ----------

# unstratified patients
phe_outlier_before = phe_outlier[phe_outlier['phe_time_before'] == True].copy()
phe_outlier_after = phe_outlier[phe_outlier['phe_time_after'] == True].copy()

# male infertility patients
mi_phe_outlier_before = mi_phe_outlier[mi_phe_outlier['phe_time_before'] == True].copy()
mi_phe_outlier_after = mi_phe_outlier[mi_phe_outlier['phe_time_after'] == True].copy()

# vasectomy patients
vas_phe_outlier_before = vas_phe_outlier[vas_phe_outlier['phe_time_before'] == True].copy()
vas_phe_outlier_after = vas_phe_outlier[vas_phe_outlier['phe_time_after'] == True].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nonoutliers

# COMMAND ----------

# unstratified patients
phe_nonoutlier_before = phe_nonoutlier[phe_nonoutlier['phe_time_before'] == True].copy()
phe_nonoutlier_after = phe_nonoutlier[phe_nonoutlier['phe_time_after'] == True].copy()

# male infertility patients
mi_phe_nonoutlier_before = mi_phe_nonoutlier[mi_phe_nonoutlier['phe_time_before'] == True].copy()
mi_phe_nonoutlier_after = mi_phe_nonoutlier[mi_phe_nonoutlier['phe_time_after'] == True].copy()

# vasectomy patients
vas_phe_nonoutlier_before = vas_phe_nonoutlier[vas_phe_nonoutlier['phe_time_before'] == True].copy()
vas_phe_nonoutlier_after = vas_phe_nonoutlier[vas_phe_nonoutlier['phe_time_after'] == True].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain mean number of phecodes per patient for unstratified, male infertility, and vasectomy patients before and after the 6-month cutoff

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outliers

# COMMAND ----------

# making sure correct number of patients for outliers
# number of all outliers should equal number of male infertility + vasectomy outliers
phe_outlier['person_id'].nunique() == (mi_phe_outlier['person_id'].nunique() + vas_phe_outlier['person_id'].nunique())

# COMMAND ----------

# before 6-month cutoff
phe_outlier_before_phe_per_pt = phe_outlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

mi_phe_outlier_before_phe_per_pt = mi_phe_outlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

vas_phe_outlier_before_phe_per_pt = vas_phe_outlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

# COMMAND ----------

# after 6-month cutoff
phe_outlier_after_phe_per_pt = phe_outlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

mi_phe_outlier_after_phe_per_pt = mi_phe_outlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

vas_phe_outlier_after_phe_per_pt = vas_phe_outlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

# COMMAND ----------

print("%%%%%%% Before 6-month cutoff %%%%%%%")
print(f"Distribution of phecodes per patient for all outliers: {phe_outlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for all outliers: {phe_outlier_before_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for male infertility outliers: {mi_phe_outlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for male infertility outliers: {mi_phe_outlier_before_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for vasectomy outliers: {vas_phe_outlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for vasectomy outliers: {vas_phe_outlier_before_phe_per_pt.mean()}\n")

# COMMAND ----------

print("%%%%%%% After 6-month cutoff %%%%%%%")
print(f"Distribution of phecodes per patient for all outliers: {phe_outlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for all outliers: {phe_outlier_after_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for male infertility outliers: {mi_phe_outlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for male infertility outliers: {mi_phe_outlier_after_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for vasectomy outliers: {vas_phe_outlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for vasectomy outliers: {vas_phe_outlier_after_phe_per_pt.mean()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nonoutliers

# COMMAND ----------

# making sure correct number of patients for nonoutliers
# number of all nonoutliers should equal number of male infertility + vasectomy nonoutliers
phe_nonoutlier['person_id'].nunique() == (mi_phe_nonoutlier['person_id'].nunique() + vas_phe_nonoutlier['person_id'].nunique())

# COMMAND ----------

# before 6-month cutoff
phe_nonoutlier_before_phe_per_pt = phe_nonoutlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

mi_phe_nonoutlier_before_phe_per_pt = mi_phe_nonoutlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

vas_phe_nonoutlier_before_phe_per_pt = vas_phe_nonoutlier_before[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

# COMMAND ----------

# after 6-month cutoff
phe_nonoutlier_after_phe_per_pt = phe_nonoutlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

mi_phe_nonoutlier_after_phe_per_pt = mi_phe_nonoutlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

vas_phe_nonoutlier_after_phe_per_pt = vas_phe_nonoutlier_after[['person_id', 'phenotype']].groupby('person_id').count().sort_values(by='phenotype', ascending=False).copy()

# COMMAND ----------

print("%%%%%%% Before 6-month cutoff %%%%%%%")
print(f"Distribution of phecodes per patient for all nonoutliers: {phe_nonoutlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for all nonoutliers: {phe_nonoutlier_before_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for male infertility nonoutliers: {mi_phe_nonoutlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for male infertility nonoutliers: {mi_phe_nonoutlier_before_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for vasectomy nonoutliers: {vas_phe_nonoutlier_before_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for vasectomy nonoutliers: {vas_phe_nonoutlier_before_phe_per_pt.mean()}\n")

# COMMAND ----------

print("%%%%%%% After 6-month cutoff %%%%%%%")
print(f"Distribution of phecodes per patient for all nonoutliers: {phe_nonoutlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for all nonoutliers: {phe_nonoutlier_after_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for male infertility nonoutliers: {mi_phe_nonoutlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for male infertility nonoutliers: {mi_phe_nonoutlier_after_phe_per_pt.mean()}\n")

print(f"Distribution of phecodes per patient for vasectomy nonoutliers: {vas_phe_nonoutlier_after_phe_per_pt.describe()}")
print(f"Mean number of phecodes per patient for vasectomy nonoutliers: {vas_phe_nonoutlier_after_phe_per_pt.mean()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mann-Whitney U test: outlier vs nonoutlier

# COMMAND ----------

print("%%%%%%% Before 6-month cutoff %%%%%%%")

# unstratified patients
statistic, p_value = mannwhitneyu(x=phe_outlier_before_phe_per_pt, y=phe_nonoutlier_before_phe_per_pt)
print(f"unstratified patients\nstatistic = {statistic}, p-value = {p_value}\n")

# male infertility patients
statistic, p_value = mannwhitneyu(x=mi_phe_outlier_before_phe_per_pt, y=mi_phe_nonoutlier_before_phe_per_pt)
print(f"male infertility patients\nstatistic = {statistic}, p-value = {p_value}\n")

# vasectomy patients
statistic, p_value = mannwhitneyu(x=vas_phe_outlier_before_phe_per_pt, y=vas_phe_nonoutlier_before_phe_per_pt)
print(f"vasectomy patients\nstatistic = {statistic}, p-value = {p_value}\n")

# COMMAND ----------

print("%%%%%%% After 6-month cutoff %%%%%%%")

# unstratified patients
statistic, p_value = mannwhitneyu(x=phe_outlier_after_phe_per_pt, y=phe_nonoutlier_after_phe_per_pt)
print(f"unstratified patients\nstatistic = {statistic}, p-value = {p_value}\n")

# male infertility patients
statistic, p_value = mannwhitneyu(x=mi_phe_outlier_after_phe_per_pt, y=mi_phe_nonoutlier_after_phe_per_pt)
print(f"male infertility patients\nstatistic = {statistic}, p-value = {p_value}\n")

# vasectomy patients
statistic, p_value = mannwhitneyu(x=vas_phe_outlier_after_phe_per_pt, y=vas_phe_nonoutlier_after_phe_per_pt)
print(f"vasectomy patients\nstatistic = {statistic}, p-value = {p_value}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

# concatentate distributions of number of phenotypes per patient

# %%%% Outliers %%%%

# **** Before 6-month cutoff ****
# unstratified patients
phe_outlier_b4_phe_per_pt_desc = phe_outlier_before_phe_per_pt.describe().transpose()
phe_outlier_b4_phe_per_pt_desc.index = ["phe_outlier_before_phe_per_pt"]
phe_outlier_b4_phe_per_pt_desc = phe_outlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# male infertility patients
mi_phe_outlier_b4_phe_per_pt_desc = mi_phe_outlier_before_phe_per_pt.describe().transpose()
mi_phe_outlier_b4_phe_per_pt_desc.index = ["mi_phe_outlier_before_phe_per_pt"]
mi_phe_outlier_b4_phe_per_pt_desc = mi_phe_outlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# vasectomy patients
vas_phe_outlier_b4_phe_per_pt_desc = vas_phe_outlier_before_phe_per_pt.describe().transpose()
vas_phe_outlier_b4_phe_per_pt_desc.index = ["vas_phe_outlier_before_phe_per_pt"]
vas_phe_outlier_b4_phe_per_pt_desc = vas_phe_outlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# **** After 6-month cutoff ****
# unstratified patients
phe_outlier_aft_phe_per_pt_desc = phe_outlier_after_phe_per_pt.describe().transpose()
phe_outlier_aft_phe_per_pt_desc.index = ["phe_outlier_after_phe_per_pt"]
phe_outlier_aft_phe_per_pt_desc = phe_outlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# male infertility patients
mi_phe_outlier_aft_phe_per_pt_desc = mi_phe_outlier_after_phe_per_pt.describe().transpose()
mi_phe_outlier_aft_phe_per_pt_desc.index = ["mi_phe_outlier_after_phe_per_pt"]
mi_phe_outlier_aft_phe_per_pt_desc = mi_phe_outlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# vasectomy patients
vas_phe_outlier_aft_phe_per_pt_desc = vas_phe_outlier_after_phe_per_pt.describe().transpose()
vas_phe_outlier_aft_phe_per_pt_desc.index = ["vas_phe_outlier_after_phe_per_pt"]
vas_phe_outlier_aft_phe_per_pt_desc = vas_phe_outlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# %%%% Nonoutliers %%%%

# **** Before 6-month cutoff ****
# unstratified patients
phe_nonoutlier_b4_phe_per_pt_desc = phe_nonoutlier_before_phe_per_pt.describe().transpose()
phe_nonoutlier_b4_phe_per_pt_desc.index = ["phe_nonoutlier_before_phe_per_pt"]
phe_nonoutlier_b4_phe_per_pt_desc = phe_nonoutlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# male infertility patients
mi_phe_nonoutlier_b4_phe_per_pt_desc = mi_phe_nonoutlier_before_phe_per_pt.describe().transpose()
mi_phe_nonoutlier_b4_phe_per_pt_desc.index = ["mi_phe_nonoutlier_before_phe_per_pt"]
mi_phe_nonoutlier_b4_phe_per_pt_desc = mi_phe_nonoutlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# vasectomy patients
vas_phe_nonoutlier_b4_phe_per_pt_desc = vas_phe_nonoutlier_before_phe_per_pt.describe().transpose()
vas_phe_nonoutlier_b4_phe_per_pt_desc.index = ["vas_phe_nonoutlier_before_phe_per_pt"]
vas_phe_nonoutlier_b4_phe_per_pt_desc = vas_phe_nonoutlier_b4_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# **** After 6-month cutoff ****
# unstratified patients
phe_nonoutlier_aft_phe_per_pt_desc = phe_nonoutlier_after_phe_per_pt.describe().transpose()
phe_nonoutlier_aft_phe_per_pt_desc.index = ["phe_nonoutlier_after_phe_per_pt"]
phe_nonoutlier_aft_phe_per_pt_desc = phe_nonoutlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# male infertility patients
mi_phe_nonoutlier_aft_phe_per_pt_desc = mi_phe_nonoutlier_after_phe_per_pt.describe().transpose()
mi_phe_nonoutlier_aft_phe_per_pt_desc.index = ["mi_phe_nonoutlier_after_phe_per_pt"]
mi_phe_nonoutlier_aft_phe_per_pt_desc = mi_phe_nonoutlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# vasectomy patients
vas_phe_nonoutlier_aft_phe_per_pt_desc = vas_phe_nonoutlier_after_phe_per_pt.describe().transpose()
vas_phe_nonoutlier_aft_phe_per_pt_desc.index = ["vas_phe_nonoutlier_after_phe_per_pt"]
vas_phe_nonoutlier_aft_phe_per_pt_desc = vas_phe_nonoutlier_aft_phe_per_pt_desc.rename({"count" : "num_of_pts"}, axis=1).copy()

# %%%% concatentate %%%%
all_desc = pd.concat([phe_outlier_b4_phe_per_pt_desc,
                      mi_phe_outlier_b4_phe_per_pt_desc,
                      vas_phe_outlier_b4_phe_per_pt_desc,
                      phe_outlier_aft_phe_per_pt_desc,
                      mi_phe_outlier_aft_phe_per_pt_desc,
                      vas_phe_outlier_aft_phe_per_pt_desc,
                      phe_nonoutlier_b4_phe_per_pt_desc,
                      mi_phe_nonoutlier_b4_phe_per_pt_desc,
                      vas_phe_nonoutlier_b4_phe_per_pt_desc,
                      phe_nonoutlier_aft_phe_per_pt_desc,
                      mi_phe_nonoutlier_aft_phe_per_pt_desc,
                      vas_phe_nonoutlier_aft_phe_per_pt_desc])

all_desc = all_desc.rename({"mean" : "mean_num_phe_per_pt"}, axis=1).copy()
all_desc_final = all_desc#.reset_index(names="mean_num_phe_per_pt").copy()

# COMMAND ----------

display(all_desc_final)

# COMMAND ----------

# save as csv
all_desc_final.to_csv("male_infertility_validation/revision_files/phe_per_pt_desc.csv")

# COMMAND ----------

# concatenate Mann-Whitney U test results
statistic, p_value = mannwhitneyu(x=phe_outlier_before_phe_per_pt, y=phe_nonoutlier_before_phe_per_pt)
phe_outlier_vs_nonoutlier_before = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["phe_outlier_vs_nonoutlier_before"])

statistic, p_value = mannwhitneyu(x=mi_phe_outlier_before_phe_per_pt, y=mi_phe_nonoutlier_before_phe_per_pt)
mi_phe_outlier_vs_nonoutlier_before = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["mi_phe_outlier_vs_nonoutlier_before"])

statistic, p_value = mannwhitneyu(x=vas_phe_outlier_before_phe_per_pt, y=vas_phe_nonoutlier_before_phe_per_pt)
vas_phe_outlier_vs_nonoutlier_before = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["vas_phe_outlier_vs_nonoutlier_before"])

statistic, p_value = mannwhitneyu(x=phe_outlier_after_phe_per_pt, y=phe_nonoutlier_after_phe_per_pt)
phe_outlier_vs_nonoutlier_after = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["phe_outlier_vs_nonoutlier_after"])

statistic, p_value = mannwhitneyu(x=mi_phe_outlier_after_phe_per_pt, y=mi_phe_nonoutlier_after_phe_per_pt)
mi_phe_outlier_vs_nonoutlier_after = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["mi_phe_outlier_vs_nonoutlier_after"])

statistic, p_value = mannwhitneyu(x=vas_phe_outlier_after_phe_per_pt, y=vas_phe_nonoutlier_after_phe_per_pt)
vas_phe_outlier_vs_nonoutlier_after = pd.DataFrame({"statistic" : [statistic[0]], "p_value" : [p_value[0]]}, 
                                         index=["vas_phe_outlier_vs_nonoutlier_after"])

all_comp = pd.concat([phe_outlier_vs_nonoutlier_before,
                      mi_phe_outlier_vs_nonoutlier_before,
                      vas_phe_outlier_vs_nonoutlier_before,
                      phe_outlier_vs_nonoutlier_after,
                      mi_phe_outlier_vs_nonoutlier_after,
                      vas_phe_outlier_vs_nonoutlier_after])

all_comp_final = all_comp#.reset_index(names="Mann_Whitney_U_comparison").copy()

# COMMAND ----------

# save as csv
all_comp_final.to_csv("male_infertility_validation/revision_files/mannwhitneyu_outlier_comp.csv")

# COMMAND ----------


