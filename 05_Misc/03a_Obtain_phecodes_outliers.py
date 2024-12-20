# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook for Step 3a: Obtain phecode diagnoses for outliers. This notebook converts icd diagnoses to phecodes.

# COMMAND ----------

import pandas as pd
from MI_Functions import *

# From https://docs.databricks.com/spark/latest/spark-sql/spark-pandas.html:
# Enable Arrow-based columnar data transfers
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' phecode notebooks

# COMMAND ----------

#phecodes_v2_spark = spark.read.format("parquet").load("male_infertility_validation/tables/phecodes/phecodes_v2")
phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

#phecodes_cat_v2_spark = spark.read.format("parquet").load("male_infertility_validation/tables/phecodes/phecodes_cat_v2")
phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent diagnoses files

# COMMAND ----------

# unstratified patients
#diag_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/icd_snomed_outlier")
#diag_outlier = diag_outlier_spark.toPandas()

diag_outlier = pd.read_pickle("male_infertility_validation/revision_files/icd_snomed_outlier.pkl")

# male infertility patients
#mi_diag_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/mi_icd_snomed_outlier")
#mi_diag_outlier = mi_diag_outlier_spark.toPandas()

mi_diag_outlier = pd.read_pickle("male_infertility_validation/revision_files/mi_icd_snomed_outlier.pkl")

# vasectomy patients
#vas_only_diag_outlier_spark = spark.read.format("parquet").load("male_infertility_validation/revision_files/vas_icd_snomed_outlier")
#vas_only_diag_outlier = vas_only_diag_outlier_spark.toPandas()

vas_only_diag_outlier = pd.read_pickle("male_infertility_validation/revision_files/vas_icd_snomed_outlier.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Obtain corresponding phecodes

# COMMAND ----------

# MAGIC %md
# MAGIC How to [select columns in PySpark](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.select.html)

# COMMAND ----------

#phecodes_v2_selected = phecodes_v2_spark.select('icd', 'phecode', 'phenotype')
#phecodes_cat_v2_selected = phecodes_cat_v2_spark.select('phecode')

phecodes_v2_selected = phecodes_v2[['icd', 'phecode', 'phenotype']]
phecodes_cat_v2_selected = phecodes_cat_v2[['phecode']]

# COMMAND ----------

# MAGIC %md
# MAGIC Reference for [join](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.join.html) and [dropDuplicates](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.dropDuplicates.html) in PySpark. Additional [reference](https://stackoverflow.com/questions/46944493/removing-duplicate-columns-after-a-df-join-in-spark) for not duplicating columns during a join

# COMMAND ----------

#phecodes_spark = phecodes_v2_selected.join(phecodes_cat_v2_selected,
#                                           ['phecode'],
#                                           'left').dropDuplicates()

phecodes = phecodes_v2_selected.merge(phecodes_cat_v2_selected,
                                      on='phecode',
                                      how='left').copy().drop_duplicates()

# COMMAND ----------

analyses = dict()

# COMMAND ----------

diags = [diag_outlier,
         mi_diag_outlier, 
         vas_only_diag_outlier]
diag_names = ['diag_outlier',
              'mi_diag_outlier', 
              'vas_only_diag_outlier']

for diag, diag_name in zip(diags, diag_names):
  print(f"Adding phecodes for {diag_name}")
  analyses[diag_name] = obtain_phecodes_stanford(icd_diag=diag, phecodes=phecodes)
  print('\n')

# COMMAND ----------

diag2_outlier = analyses['diag_outlier'].copy()
mi_diag2_outlier = analyses['mi_diag_outlier'].copy()
vas_only_diag2_outlier = analyses['vas_only_diag_outlier'].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add in first phenotype start date for each patient and whether it was before or after analysis cutoff date

# COMMAND ----------

# unstratified patients
first_phe_date = add_first_phe_date(df=diag2_outlier)
diag2_outlier_final = add_phe_rel_date(df=diag2_outlier, first_phe_date=first_phe_date, cutoff_date='analysis_cutoff_date')

# patients with male infertility
mi_first_phe_date = add_first_phe_date(df=mi_diag2_outlier)
mi_diag2_outlier_final = add_phe_rel_date(df=mi_diag2_outlier, first_phe_date=mi_first_phe_date, cutoff_date='analysis_cutoff_date')

# patients who have underwent a vasectomy related procedure
vas_only_first_phe_date = add_first_phe_date(df=vas_only_diag2_outlier)
vas_only_diag2_outlier_final = add_phe_rel_date(df=vas_only_diag2_outlier, first_phe_date=vas_only_first_phe_date, cutoff_date='analysis_cutoff_date')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert back to spark DataFrame and save

# COMMAND ----------

# converting back to spark
#diag_outlier_final_spark = spark.createDataFrame(diag2_outlier_final)
#mi_diag_outlier_final_spark = spark.createDataFrame(mi_diag2_outlier_final)
#vas_only_diag_outlier_final_spark = spark.createDataFrame(vas_only_diag2_outlier_final)

# save
diag2_outlier_final.to_pickle("male_infertility_validation/revision_files/phe_outlier.pkl")
mi_diag2_outlier_final.to_pickle("male_infertility_validation/revision_files/mi_phe_outlier.pkl")
vas_only_diag2_outlier_final.to_pickle("male_infertility_validation/revision_files/vas_phe_outlier.pkl")

# COMMAND ----------


