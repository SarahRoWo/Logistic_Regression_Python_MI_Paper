# Databricks notebook source
# MAGIC %md
# MAGIC ## Obtain quartiles for delta between diagnoses and first mi diagnosis (or vasectomy record) before and after analysis cutoff date

# COMMAND ----------

# MAGIC %md
# MAGIC ## 'Import' functions

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in pertinent diagnosis profiles

# COMMAND ----------

# Patients with male infertility
mi_phe = pd.read_pickle("male_infertility_validation/revision_files/mi_phe_w_diag_deltas.pkl")

# Vasectomy only patients
vas_only_phe = pd.read_pickle("male_infertility_validation/revision_files/vas_phe_w_diag_deltas.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter for phenotypes before and after male infertility diagnosis or vasectomy procedure

# COMMAND ----------

# Before
mi_phe_before = mi_phe[mi_phe['phe_time_before'] == 1]
vas_only_phe_before = vas_only_phe[vas_only_phe['phe_time_before'] == 1]

# After
mi_phe_after = mi_phe[mi_phe['phe_time_after'] == 1]
vas_only_phe_after = vas_only_phe[vas_only_phe['phe_time_after'] == 1]

# COMMAND ----------

# Only keep phenotypes that overlap for male infertility and control patients

# Before
phenotypes_both_before = set(mi_phe_before['phenotype'].unique()) & set(vas_only_phe_before['phenotype'].unique())

# After
phenotypes_both_after = set(mi_phe_after['phenotype'].unique()) & set(vas_only_phe_after['phenotype'].unique())

# COMMAND ----------

len(phenotypes_both_before)

# COMMAND ----------

len(phenotypes_both_after)

# COMMAND ----------

# remove None if in phentoypes_both
print(f"None value in phenotypes_both_before: {None in phenotypes_both_before}")
print(f"None value in phenotypes_both_after: {None in phenotypes_both_after}")

# COMMAND ----------

# Combine diagnoses and only keep the ones represented in both mi and vas patients

# Add whether patients have male infertility

# Before
mi_phe_before2 = mi_phe_before.copy()
mi_phe_before2['has_mi'] = 1

vas_only_phe_before2 = vas_only_phe_before.copy()
vas_only_phe_before2['has_mi'] = 0

# After
mi_phe_after2 = mi_phe_after.copy()
mi_phe_after2['has_mi'] = 1

vas_only_phe_after2 = vas_only_phe_after.copy()
vas_only_phe_after2['has_mi'] = 0

# Combine diagnoses

# Prior to removing non overlapping diagnoses
diag_combined_before_pre = pd.concat([mi_phe_before2, vas_only_phe_before2])
diag_combined_after_pre = pd.concat([mi_phe_after2, vas_only_phe_after2])

# After removing non overlapping diagnoses
diag_combined_before = diag_combined_before_pre[diag_combined_before_pre['phenotype'].isin(phenotypes_both_before)].copy()
diag_combined_after = diag_combined_after_pre[diag_combined_after_pre['phenotype'].isin(phenotypes_both_after)].copy()

# COMMAND ----------

# Group by phenotype and describe the distribution of diag_cutoff_delta_approx_m for distribution_before
distribution_before = diag_combined_before.groupby('phenotype')['diag_cutoff_delta_approx_m'].describe()

distribution_before.head()

# COMMAND ----------

# Drop count column
distribution_before_censored = distribution_before.drop('count', axis=1).reset_index().copy()

distribution_before_censored.head()

# COMMAND ----------

# Group by phenotype and describe the distribution of diag_cutoff_delta_approx_m for distribution_after
distribution_after = diag_combined_after.groupby('phenotype')['diag_cutoff_delta_approx_m'].describe()

distribution_after.head()

# COMMAND ----------

# Drop count column
distribution_after_censored = distribution_after.drop('count', axis=1).reset_index().copy()

distribution_after_censored.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of median delta of diagnoses (in months)

# COMMAND ----------

distribution_before_censored['50%'].describe()

# COMMAND ----------

distribution_after_censored['50%'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save distribution of diagnosis times in months

# COMMAND ----------

distribution_before_censored.to_pickle("male_infertility_validation/revision_files/fx_diag_delta_dist_b4_cutoff_m_c.pkl")
distribution_after_censored.to_pickle("male_infertility_validation/revision_files/fx_diag_delta_dist_aft_cutoff_m_c.pkl")

# COMMAND ----------

