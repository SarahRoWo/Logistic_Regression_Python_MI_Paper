# Databricks notebook source
import pandas as pd
import numpy as np
import functools as ft
import datetime

# COMMAND ----------

# MAGIC %run male_infertility_validation/MI_Functions.py

# COMMAND ----------

# DBTITLE 1,Load in notebooks
# Patients
mi_person = pd.read_csv("male_infertility_validation/raw_data/person_case.csv",sep = "\t")

vas_person = pd.read_csv("male_infertility_validation/raw_data/person_control.csv",sep = "\t")

# Phecodes
phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")

phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

phecodes_v2_selected = phecodes_v2[['icd', 'phecode', 'phenotype']].copy()
phecodes_cat_v2_selected = phecodes_cat_v2['phecode'].to_frame().copy()

phecodes = phecodes_v2_selected.merge(phecodes_cat_v2_selected,
                                           on='phecode',
                                           how='left').copy().drop_duplicates()

# Tables
pt_cond = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_control.csv", sep = "\t")

pt_obs = pd.read_csv("male_infertility_validation/raw_data/observation_control.csv",sep = "\t")

pt_meas = pd.read_csv("male_infertility_validation/raw_data/measurement_control.csv",sep = "\t")

pt_proc = pd.read_csv("male_infertility_validation/raw_data/procedure_occurrence_control.csv",sep = "\t")

pt_visit = pd.read_csv("male_infertility_validation/raw_data/visit_occurrence_control.csv",sep = "\t")

concepts = pd.read_csv("male_infertility_validation/raw_data/concepts.csv")

# Vasectomy concepts
vasectomy_concepts = pd.read_csv("male_infertility_validation/concepts/vasectomy_concepts.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Filter out patients who have had a male infertility diagnosis

# COMMAND ----------

print(f"Number of vasectomy patients before filtering out those who have a male infertility diagnosis: {vas_person.shape[0]}")
vas_temp_1 = vas_person[~vas_person['person_id'].isin(mi_person['person_id'])].copy()
#3340

print(f"Number of vasectomy patients after filtering out those who have a male infertility diagnosis: {vas_temp_1.shape[0]}")
#3127
# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filter out patients who are not male-identified

# COMMAND ----------

vas_temp_2 = vas_temp_1[vas_temp_1['gender_concept_id'] == 8507].copy()
print(f"Number of male infertility patients after selecting only those who are male-identified: {vas_temp_2.shape[0]}")
#3121
# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter out patients who are under the age of 18

# COMMAND ----------

# Make estimated_age column

# Get current year
today = datetime.date.today()
current_year = today.year

vas_temp_2['estimated_age'] = current_year - vas_temp_2['year_of_birth']

# COMMAND ----------

vas_temp_3 = vas_temp_2[vas_temp_2['estimated_age'] >= 18].copy()
print(f"Number of male infertility patients that are 18+: {vas_temp_3.shape[0]}")
#3121
# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Filter out patients who have no  phenotypes

# COMMAND ----------

# Only retrieve conditions from vasectomy patients
pt_cond_icd_vas = pt_cond.merge(vas_temp_3['person_id'].to_frame(), on='person_id').drop_duplicates()
pt_cond_icd_vas.shape

# COMMAND ----------

# Inner join with phecodes, which will remove diagnoses without phecode-corresponding phenotypes
pt_cond_icd_vas_2 = pt_cond_icd_vas.merge(phecodes, left_on='condition_source_value', right_on='icd').drop_duplicates()
pt_cond_icd_vas_2.shape

# COMMAND ----------

# Patients after filtering out those with no phenotypes
vas_temp_4 = vas_temp_3[vas_temp_3['person_id'].isin(pt_cond_icd_vas_2['person_id'])]

# Confirming patient demographics table has one row for each patient
vas_temp_4.shape[0] == vas_temp_4['person_id'].nunique()

# COMMAND ----------

# Number of patients
print(f"Number of patients after filtering for at least one phenotype: {vas_temp_4['person_id'].nunique()}")
#2655
# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add total number of visits

# COMMAND ----------

# Visit occurrence for patients with vasectomy
pt_visit_vas = pt_visit[pt_visit['person_id'].isin(vas_temp_4['person_id'])].drop_duplicates()
pt_visit_vas.shape

# COMMAND ----------

# Total number of visits per vasectomy patient
temp_1 = pt_visit_vas[['person_id', 'visit_occurrence_id']].groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_total'}, axis=1)
temp_1.head(3)

# COMMAND ----------

# Merge with vas_temp_4
vas_temp_5 = vas_temp_4.merge(temp_1, on='person_id')

# COMMAND ----------

vas_temp_5.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Obtain total length in the EMR in months

# COMMAND ----------

# Obtain all visit start dates for each vasectomy patient
vas_first_vo = pt_visit_vas[['person_id', 'visit_start_datetime']].groupby('person_id').min().rename({'visit_start_datetime' : 'first_visit_date'}, axis=1)
vas_first_vo.head(3)

# COMMAND ----------

# Making sure no NA values
vas_first_vo[vas_first_vo['first_visit_date'].isna()].shape[0]

# COMMAND ----------

# Obtain all visit end dates for each vasectomy patient
vas_last_vo = pt_visit_vas[['person_id', 'visit_end_datetime']].groupby('person_id').max().rename({'visit_end_datetime' : 'last_visit_date'}, axis=1)
vas_last_vo.head(3)

# COMMAND ----------

# Making sure no NA values
vas_last_vo[vas_last_vo['last_visit_date'].isna()].shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get length in the EMR as months
# MAGIC [Reference](https://statisticsglobe.com/convert-timedelta-months-python) for extracting months

# COMMAND ----------

vas_first_last_vo = vas_first_vo.merge(vas_last_vo, on='person_id')
vas_first_last_vo['emr_months_total'] = vas_first_last_vo['last_visit_date'].dt.to_period('M').astype(int) - \
                                        vas_first_last_vo['first_visit_date'].dt.to_period('M').astype(int)


# COMMAND ----------

vas_first_last_vo.head(3)

# COMMAND ----------

# Add in number of months in the EMR total
vas_temp_6 = vas_temp_5.merge(vas_first_last_vo['emr_months_total'], left_on='person_id', right_index=True)

# COMMAND ----------

vas_temp_6.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Get first vasectomy-related procedure

# COMMAND ----------

vas_temp_6['person_id'].nunique()

# COMMAND ----------

# %%% Not found at UCSF %%%
# Conditions for remaining vasectomy patients
pt_cond_vas = pt_cond[pt_cond['person_id'].isin(vas_temp_6['person_id'])]

# Observations for remaining vasectomy patients
pt_obs_vas = pt_obs[pt_obs['person_id'].isin(vas_temp_6['person_id'])]

# %%% Found at UCSF %%%
# Measurements for remaining vasectomy patients
pt_meas_vas = pt_meas[pt_meas['person_id'].isin(vas_temp_6['person_id'])]

# Procedures for remaining vasectomy patients
pt_proc_vas = pt_proc[pt_proc['person_id'].isin(vas_temp_6['person_id'])]

# COMMAND ----------

# %%% Not found at UCSF %%%
# Get vasectomy-related conditions
pt_cond_vas_cond_vas = pt_cond_vas[pt_cond_vas['condition_concept_id'].isin(vasectomy_concepts['concept_id'])].drop_duplicates()

# Get vasectomy-related observations
pt_obs_vas_obs_vas = pt_obs_vas[pt_obs_vas['observation_concept_id'].isin(vasectomy_concepts['concept_id'])].drop_duplicates()

# Get vasectomy-related measurement values
pt_meas_val_vas_meas_val_vas = pt_meas_vas[pt_meas_vas['value_as_concept_id'].isin(vasectomy_concepts['concept_id'])].drop_duplicates()

# %%% Found at UCSF %%%
# Get vasectomy-related measurements
pt_meas_vas_meas_vas = pt_meas_vas[pt_meas_vas['measurement_concept_id'].isin(vasectomy_concepts['concept_id'])].drop_duplicates()

# Get vasectomy-related procedures
pt_proc_vas_proc_vas = pt_proc_vas[pt_proc_vas['procedure_concept_id'].isin(vasectomy_concepts['concept_id'])].drop_duplicates()

# COMMAND ----------

pt_meas_vas_meas_vas.head(3)

# COMMAND ----------

# %%% Not Found at UCSF %%%
# First vasectomy-related condition date
pt_cond_vas_cond_vas_first = pt_cond_vas_cond_vas[['person_id', 'condition_start_date']].groupby('person_id').min()

# First vasectomy-related observation date
pt_obs_vas_obs_vas_first = pt_obs_vas_obs_vas[['person_id', 'observation_date']].groupby('person_id').min()

# First vasectomy-related measurement value date
pt_meas_val_vas_meas_val_vas_first = pt_meas_val_vas_meas_val_vas[['person_id', 'measurement_date']].groupby('person_id').min()
pt_meas_val_vas_meas_val_vas_first = pt_meas_val_vas_meas_val_vas_first.rename({'measurement_date' : 'meas_val_date'}, axis=1)

# %%% Found at UCSF %%%
# First vasectomy-related measurement date
pt_meas_vas_meas_vas_first = pt_meas_vas_meas_vas[['person_id', 'measurement_date']].groupby('person_id').min()

# First vasectomy-related procedure date
pt_proc_vas_proc_vas_first = pt_proc_vas_proc_vas[['person_id', 'procedure_date']].groupby('person_id').min()

# Merge
#pt_vas_first = pt_proc_vas_proc_vas_first.merge(pt_meas_vas_meas_vas_first, on='person_id', how='outer')

first_date_dfs = [pt_proc_vas_proc_vas_first,
                  pt_meas_vas_meas_vas_first, 
                  pt_cond_vas_cond_vas_first, 
                  pt_obs_vas_obs_vas_first, 
                  pt_meas_val_vas_meas_val_vas_first]

pt_vas_first = ft.reduce(lambda left, right: pd.merge(left, right, on='person_id', how='outer'), first_date_dfs)

# COMMAND ----------

pt_vas_first

# COMMAND ----------

# Convert columns to datetime
pt_vas_first['procedure_date'] = pd.to_datetime(pt_vas_first['procedure_date'], format='%Y-%m-%d')
pt_vas_first['measurement_date'] = pd.to_datetime(pt_vas_first['measurement_date'], format='%Y-%m-%d')
pt_vas_first['condition_start_date'] = pd.to_datetime(pt_vas_first['condition_start_date'], format='%Y-%m-%d')
pt_vas_first['observation_date'] = pd.to_datetime(pt_vas_first['observation_date'], format='%Y-%m-%d')
pt_vas_first['meas_val_date'] = pd.to_datetime(pt_vas_first['meas_val_date'], format='%Y-%m-%d')

# COMMAND ----------

# Get earlier date for first vasectomy-related procedure or measurement date
pt_vas_first['first_mi_or_vas_date'] = pt_vas_first[['procedure_date', 
                                                     'measurement_date',
                                                     'condition_start_date',
                                                     'observation_date',
                                                     'meas_val_date']].min(axis=1)

# COMMAND ----------

# Make sure number of patients are the same in vas_temp_6 and pt_vas_first
pt_vas_first.shape[0] == vas_temp_6['person_id'].nunique()

# COMMAND ----------

# Merge to demographics table
vas_temp_7 = vas_temp_6.merge(pt_vas_first['first_mi_or_vas_date'].to_frame(), left_on='person_id', right_index=True)

# COMMAND ----------

vas_temp_7.head(3)

# COMMAND ----------

# Add 6 months cutoff date
vas_temp_7['analysis_cutoff_date'] = vas_temp_7['first_mi_or_vas_date'] + pd.DateOffset(months=6)

# COMMAND ----------

vas_temp_7.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Add estimated age of first vasectomy-related procedure or measurement

# COMMAND ----------

# Convert year of birth to datetime
vas_temp_7['year_of_birth'] = pd.to_datetime(vas_temp_7['year_of_birth'], format='%Y')

# Obtain estimated age
vas_temp_7['mi_or_vas_est_age'] = pd.DatetimeIndex(vas_temp_7['first_mi_or_vas_date']).year - \
                                      pd.DatetimeIndex(vas_temp_7['year_of_birth']).year

# COMMAND ----------

vas_temp_7.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Filter out patients who have cutoff data after or at same time as refresh date

# COMMAND ----------

# MAGIC %md
# MAGIC Obtain [last day of previous month](https://stackoverflow.com/questions/9724906/python-date-of-the-previous-month)

# COMMAND ----------

today = datetime.date.today()
first_day_of_month = today.replace(day=1)
last_month = first_day_of_month - datetime.timedelta(days=1)

last_month_formatted = last_month.strftime("%Y-%m-%d")

# Because running before August refresh...
last_month_formatted = datetime.datetime.strptime("2023-06-30", "%Y-%m-%d")

# COMMAND ----------

vas_temp_8 = vas_temp_7[vas_temp_7['analysis_cutoff_date'] < last_month_formatted]

# Number of patients should remain this:
print(f"Number of patients {vas_temp_8['person_id'].nunique()}")

print(f"Each row corresponds to a unique patient: {vas_temp_8.shape[0] == vas_temp_8['person_id'].nunique()} ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Obtain number of visits before, same as, and after cutoff date

# COMMAND ----------

# Add cutoff date to visit occurrence table for patients with vasectomy
pt_visit_vas_2 = pt_visit_vas.merge(vas_temp_8[['person_id', 'analysis_cutoff_date']], on='person_id').drop_duplicates()

# COMMAND ----------

# Before analysis cutoff date
pt_visit_vas_before = pt_visit_vas_2[pt_visit_vas_2['visit_end_date'] < pt_visit_vas_2['analysis_cutoff_date']]

# Same as analysis cutoff date
pt_visit_vas_same = pt_visit_vas_2[pt_visit_vas_2['visit_start_date'] == pt_visit_vas_2['analysis_cutoff_date']]

# After analysis cutoff date
pt_visit_vas_after = pt_visit_vas_2[pt_visit_vas_2['visit_start_date'] > pt_visit_vas_2['analysis_cutoff_date']]

# COMMAND ----------

# Number of visits before analysis cutoff date
num_visits_before = pt_visit_vas_before[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_before'}, axis=1)

# Number of visits during analysis cutoff date
num_visits_same = pt_visit_vas_same[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_same'}, axis=1)

# Number of visits after analysis cutoff date
num_visits_after = pt_visit_vas_after[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_after'}, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC [Reference](https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns) for merging multiple dfs

# COMMAND ----------

# Merge number of visits before, during, and after analysis cutoff date to dataframe
num_visits_dfs = [vas_temp_8, num_visits_before, num_visits_same, num_visits_after]

vas_temp_9 = ft.reduce(lambda left, right: pd.merge(left, right, on='person_id', how='left'), num_visits_dfs).fillna(0)

# COMMAND ----------

vas_temp_9.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Obtain length in the EMR in months before, during, and after analysis cutoff date

# COMMAND ----------

# %%%% Before analysis cutoff date %%%%
# First visit start date
temp_1 = pt_visit_vas_before[['person_id', 'visit_start_date']].drop_duplicates().groupby('person_id').min().rename({'visit_start_date' : 'first_visit_start_date'}, axis=1)

# Last visit end date
temp_2 = pt_visit_vas_before[['person_id', 'visit_end_date']].drop_duplicates().groupby('person_id').max().rename({'visit_end_date' : 'last_visit_end_date'}, axis=1)

# Merge
temp_3 = temp_1.merge(temp_2, on='person_id')
temp_3['first_visit_start_date'] = pd.to_datetime(temp_3['first_visit_start_date'], format="%Y-%m-%d")
temp_3['last_visit_end_date'] = pd.to_datetime(temp_3['last_visit_end_date'], format="%Y-%m-%d")

# Length in months
temp_3['emr_months_before'] = temp_3['last_visit_end_date'].dt.to_period('M').astype(int) - \
                              temp_3['first_visit_start_date'].dt.to_period('M').astype(int)

# Only retrieve emr_months_before column
emr_months_before = temp_3['emr_months_before'].to_frame()

# %%%% After analysis cutoff date %%%%
# First visit start date
temp_1 = pt_visit_vas_after[['person_id', 'visit_start_date']].drop_duplicates().groupby('person_id').min().rename({'visit_start_date' : 'first_visit_start_date'}, axis=1)

# Last visit end date
temp_2 = pt_visit_vas_after[['person_id', 'visit_end_date']].drop_duplicates().groupby('person_id').max().rename({'visit_end_date' : 'last_visit_end_date'}, axis=1)

# Merge
temp_3 = temp_1.merge(temp_2, on='person_id')
temp_3['first_visit_start_date'] = pd.to_datetime(temp_3['first_visit_start_date'], format="%Y-%m-%d")
temp_3['last_visit_end_date'] = pd.to_datetime(temp_3['last_visit_end_date'], format="%Y-%m-%d")

# Length in months
temp_3['emr_months_after'] = temp_3['last_visit_end_date'].dt.to_period('M').astype(int) - \
                             temp_3['first_visit_start_date'].dt.to_period('M').astype(int)

# Only retrieve emr_months_after column
emr_months_after = temp_3['emr_months_after'].to_frame()

# COMMAND ----------

# Merge length in the EMR to dataframe
emr_months_dfs = [vas_temp_9, emr_months_before, emr_months_after]

vas_temp_10 = ft.reduce(lambda left, right: pd.merge(left, right, on='person_id', how='left'), emr_months_dfs).fillna(0)

# COMMAND ----------

vas_temp_10.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Add race, ethnicity, gender values and prune unneccessary columns

# COMMAND ----------

# Add concept_name for gender_concept_id
vas_temp_11 = vas_temp_10.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='gender_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'gender'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Add concept_name for race_concept_id
vas_temp_12 = vas_temp_11.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='race_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'race'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Add concept_name for ethnicity_concept_id
vas_temp_13 = vas_temp_12.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='ethnicity_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'ethnicity'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Retrieve columns
list(vas_temp_13.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC Keeping the following columns:
# MAGIC ```
# MAGIC ['person_id',
# MAGIC  'year_of_birth',
# MAGIC  'estimated_age',
# MAGIC  'gender_concept_id',
# MAGIC  'gender',
# MAGIC  'race_concept_id',
# MAGIC  'race',
# MAGIC  'ethnicity_concept_id',
# MAGIC  'ethnicity',
# MAGIC  'location_source_value',
# MAGIC  'adi',
# MAGIC  'adi_category',
# MAGIC  'num_visits_total',
# MAGIC  'emr_months_total',
# MAGIC  'first_mi_or_vas_date',
# MAGIC  'analysis_cutoff_date',
# MAGIC  'mi_or_vas_est_age',
# MAGIC  'num_visits_before',
# MAGIC  'num_visits_same',
# MAGIC  'num_visits_after',
# MAGIC  'emr_months_before',
# MAGIC  'emr_months_after']
# MAGIC ```

# COMMAND ----------

cols_to_keep = [
    "person_id",
    "year_of_birth",
    "estimated_age",
    "gender_concept_id",
    "gender",
    "race_concept_id",
    "race",
    "ethnicity_concept_id",
    "ethnicity",
    "num_visits_total",
    "emr_months_total",
    "first_mi_or_vas_date",
    "analysis_cutoff_date",
    "mi_or_vas_est_age",
    "num_visits_before",
    "num_visits_same",
    "num_visits_after",
    "emr_months_before",
    "emr_months_after",
]

# COMMAND ----------

vas_temp_14 = vas_temp_13[cols_to_keep].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Check and save

# COMMAND ----------

# Check
print(f"Number of patients stay the same after refresh filtering: {vas_temp_10.shape[0] == vas_temp_8.shape[0]}")

# COMMAND ----------

vas_temp_14.shape[0]

# COMMAND ----------

# Save
vas_temp_14.to_csv("male_infertility_validation/demographics/vas_pts_only_final.csv")
vas_temp_14.to_pickle("male_infertility_validation/demographics/vas_pts_only_final.pkl")

# COMMAND ----------

