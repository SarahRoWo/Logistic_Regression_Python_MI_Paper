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
mi_person = pd.read_csv("male_infertility_validation/raw_data/person_case.csv", sep="\t")

vas_person = pd.read_csv("male_infertility_validation/raw_data/person_control.csv", sep="\t")

# Phecodes
phecodes_cat_v2 = pd.read_csv("male_infertility_validation/phecodes/cat_phecodes_v2.csv")

phecodes_v2 = pd.read_csv("male_infertility_validation/phecodes/all_phecodes_v2.csv")

phecodes_v2_selected = phecodes_v2[['icd', 'phecode', 'phenotype']].copy()
phecodes_cat_v2_selected = phecodes_cat_v2['phecode'].to_frame().copy()

phecodes = phecodes_v2_selected.merge(phecodes_cat_v2_selected,
                                           on='phecode',
                                           how='left').copy().drop_duplicates()

# Tables
pt_cond = pd.read_csv("male_infertility_validation/raw_data/condition_occurrence_case.csv", sep="\t")

pt_obs = pd.read_csv("male_infertility_validation/raw_data/observation_case.csv", sep="\t")

pt_proc = pd.read_csv("male_infertility_validation/raw_data/procedure_occurrence_case.csv", sep="\t")

pt_visit = pd.read_csv("male_infertility_validation/raw_data/visit_occurrence_case.csv", sep="\t")

concepts = pd.read_csv("male_infertility_validation/raw_data/concepts.csv")


### change column name 
def convert_columns_to_lowercase(df):
    new_column_names = {col: col.lower() for col in df.columns}
    df.rename(columns=new_column_names, inplace=True)
    return df

mi_person = convert_columns_to_lowercase(mi_person)
vas_person = convert_columns_to_lowercase(vas_person)
pt_cond = convert_columns_to_lowercase(pt_cond)
pt_obs= convert_columns_to_lowercase(pt_obs)
pt_proc = convert_columns_to_lowercase(pt_proc)
pt_visit = convert_columns_to_lowercase(pt_visit)
concepts= convert_columns_to_lowercase(concepts)


# Male infertility concepts
male_infertility_concepts = pd.read_csv("male_infertility_validation/concepts/male_infertility_concepts.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Filter out patients who have had a vasectomy

# COMMAND ----------

print(f"Number of male infertility patients before filtering out those who have had vasectomy: {pt_person.shape[0]}")
mi_temp_1 = mi_person[~mi_person['person_id'].isin(vas_person['person_id'])].copy()

## Stanford data result: 9909

print(f"Number of male infertility patients after filtering out those who have had vasectomy: {mi_temp_1.shape[0]}")

## Stanford data result: 9696

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Filter out patients who are not male-identified

# COMMAND ----------

mi_temp_2 = mi_temp_1[mi_temp_1['gender_concept_id'] == 8507].copy()
print(f"Number of male infertility patients after selecting only those who are male-identified: {mi_temp_2.shape[0]}")


## Stanford 8212

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filter out patients who are under the age of 18

# COMMAND ----------

# Make estimated_age column

# Get current year
today = datetime.date.today()
current_year = today.year

mi_temp_2['estimated_age'] = current_year - mi_temp_2['year_of_birth']

# COMMAND ----------

mi_temp_3 = mi_temp_2[mi_temp_2['estimated_age'] >= 18].copy()
print(f"Number of male infertility patients that are 18+: {mi_temp_3.shape[0]}")

#stanford:8205

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Filter out patients who have no non male infertility phenotypes

# COMMAND ----------

# ICD concepts related to male infertility that may be mapped to phecodes
temp_1 = pt_cond[['condition_concept_id', 'condition_source_value']].copy()
temp_2 = temp_1[temp_1['condition_concept_id'].isin(male_infertility_concepts['concept_id'])].drop_duplicates()
mi_icd = temp_2.rename({'condition_concept_id' : 'snomed_concept_id',
                        'condition_source_value' : 'icd_concept_code'},
                        axis=1)
mi_icd.shape

# COMMAND ----------

# Map male infertility related ICD codes to phecodes
# ICD concept code 606 for male infertility does not have phecode-corresponding phenotype
temp_1 = mi_icd.merge(phecodes[['icd', 'phecode', 'phenotype']], left_on='icd_concept_code', right_on='icd', how='left').drop_duplicates()
mi_phe = temp_1.drop('icd', axis=1)
mi_phe.shape

# COMMAND ----------

# Get diagnoses of male infertility patients that are not male infertility related

# COMMAND ----------

# Only retrieve conditions from male infertility patients
pt_cond_icd_mi = pt_cond.merge(mi_temp_3['person_id'].to_frame(), on='person_id').drop_duplicates()
pt_cond_icd_mi.shape

# COMMAND ----------

# Diagnoses that are not mapped to male infertility concept code
pt_cond_icd_mi_2 = pt_cond_icd_mi[~pt_cond_icd_mi['condition_source_value'].isin(mi_phe['icd_concept_code'])].drop_duplicates()
pt_cond_icd_mi_2.shape

# COMMAND ----------

# Inner join with phecodes, which will remove diagnoses without phecode-corresponding phenotypes
pt_cond_icd_mi_3 = pt_cond_icd_mi_2.merge(phecodes, left_on='condition_source_value', right_on='icd').drop_duplicates()
pt_cond_icd_mi_3.shape

# COMMAND ----------

# Patients after filtering out those with no male infertility corresponding phenotype
mi_temp_4 = mi_temp_3[mi_temp_3['person_id'].isin(pt_cond_icd_mi_3['person_id'])]

# Confirming patient demographics table has one row for each patient
mi_temp_4.shape[0] == mi_temp_4['person_id'].nunique()

# COMMAND ----------

# Number of patients
print(f"Number of patients after filtering for at least one non male infertility diagnosis: {mi_temp_4['person_id'].nunique()}")

## 5672

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Add total number of visits

# COMMAND ----------

# Visit occurrence for patients with male infertility
pt_visit_mi = pt_visit[pt_visit['person_id'].isin(mi_temp_4['person_id'])].drop_duplicates()
pt_visit_mi.shape

# COMMAND ----------

# Total number of visits per male infertility patient
temp_1 = pt_visit_mi[['person_id', 'visit_occurrence_id']].groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_total'}, axis=1)
temp_1.head(3)

# COMMAND ----------

# Merge with mi_temp_4
mi_temp_5 = mi_temp_4.merge(temp_1, on='person_id')

# COMMAND ----------

mi_temp_5.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Obtain total length in the EMR in months

# COMMAND ----------

# Obtain all visit start dates for each male infertility patient
mi_first_vo = pt_visit_mi[['person_id', 'visit_start_datetime']].groupby('person_id').min().rename({'visit_start_datetime' : 'first_visit_date'}, axis=1)
mi_first_vo.head(3)

# COMMAND ----------

# Making sure no NA values
mi_first_vo[mi_first_vo['first_visit_date'].isna()].shape[0]

# COMMAND ----------

# Obtain all visit end dates for each male infertility patient
mi_last_vo = pt_visit_mi[['person_id', 'visit_end_datetime']].groupby('person_id').max().rename({'visit_end_datetime' : 'last_visit_date'}, axis=1)
mi_last_vo.head(3)

# COMMAND ----------

# Making sure no NA values
mi_last_vo[mi_last_vo['last_visit_date'].isna()].shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get length in the EMR as months
# MAGIC [Reference](https://statisticsglobe.com/convert-timedelta-months-python) for extracting months

# COMMAND ----------

mi_first_last_vo = mi_first_vo.merge(mi_last_vo, on='person_id')

mi_first_last_vo['last_visit_date'] = pd.to_datetime(mi_first_last_vo['last_visit_date'])
mi_first_last_vo['first_visit_date'] = pd.to_datetime(mi_first_last_vo['first_visit_date'])


mi_first_last_vo['emr_months_total'] = mi_first_last_vo['last_visit_date'].dt.to_period('M').astype(int) - \
                                       mi_first_last_vo['first_visit_date'].dt.to_period('M').astype(int)


# COMMAND ----------

mi_first_last_vo.head(3)

# COMMAND ----------

# Add in number of months in the EMR total
mi_temp_6 = mi_temp_5.merge(mi_first_last_vo['emr_months_total'], left_on='person_id', right_index=True)

# COMMAND ----------

mi_temp_6.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Get first male infertility diagnosis

# COMMAND ----------

# Conditions for remaining male infertility patients
pt_cond_mi = pt_cond[pt_cond['person_id'].isin(mi_temp_6['person_id'])]

# COMMAND ----------

pt_cond_mi_diag_mi = pt_cond_mi[pt_cond_mi['condition_concept_id'].isin(male_infertility_concepts['concept_id'])].drop_duplicates()

# COMMAND ----------

# First male infertility diagnosis date
pt_cond_mi_diag_mi_first = pt_cond_mi_diag_mi[['person_id', 'condition_start_date']].groupby('person_id').min()

# COMMAND ----------

# Merge to demographics table
mi_temp_7 = mi_temp_6.merge(pt_cond_mi_diag_mi_first, left_on='person_id', right_index=True).rename({'condition_start_date' : 'first_mi_or_vas_date'}, axis=1)

# COMMAND ----------

mi_temp_7.head(3)

# COMMAND ----------

mi_temp_7['first_mi_or_vas_date']=pd.to_datetime( mi_temp_7['first_mi_or_vas_date'])
# Add 6 months cutoff date
mi_temp_7['analysis_cutoff_date'] = mi_temp_7['first_mi_or_vas_date'] + pd.DateOffset(months = 6)

# COMMAND ----------

mi_temp_7.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Add estimated age of first male infertility diagnosis

# COMMAND ----------

# Convert year of birth to datetime
mi_temp_7['year_of_birth'] = pd.to_datetime(mi_temp_7['year_of_birth'], format='%Y')

# Obtain estimated age
mi_temp_7['mi_or_vas_est_age'] = pd.DatetimeIndex(mi_temp_7['first_mi_or_vas_date']).year - \
                                     pd.DatetimeIndex(mi_temp_7['year_of_birth']).year

# COMMAND ----------

mi_temp_7.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Filter out patients who have cutoff data after or at same time as refresh date

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

mi_temp_8 = mi_temp_7[mi_temp_7['analysis_cutoff_date'] < last_month_formatted]

# Number of patients should remain this:
print(f"Number of patients {mi_temp_8['person_id'].nunique()}")

print(f"Each row corresponds to a unique patient: {mi_temp_8.shape[0] == mi_temp_8['person_id'].nunique()} ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Obtain number of visits before, same as, and after cutoff date

# COMMAND ----------

# Add cutoff date to visit occurrence table for patients with male infertility
pt_visit_mi_2 = pt_visit_mi.merge(mi_temp_8[['person_id', 'analysis_cutoff_date']], on='person_id').drop_duplicates()

# COMMAND ----------

# Before analysis cutoff date
pt_visit_mi_before = pt_visit_mi_2[pt_visit_mi_2['visit_end_date'] < pt_visit_mi_2['analysis_cutoff_date']]

# Same as analysis cutoff date
pt_visit_mi_same = pt_visit_mi_2[pt_visit_mi_2['visit_start_date'] == pt_visit_mi_2['analysis_cutoff_date']]

# After analysis cutoff date
pt_visit_mi_after = pt_visit_mi_2[pt_visit_mi_2['visit_start_date'] > pt_visit_mi_2['analysis_cutoff_date']]

# COMMAND ----------

# Number of visits before analysis cutoff date
num_visits_before = pt_visit_mi_before[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_before'}, axis=1)

# Number of visits during analysis cutoff date
num_visits_same = pt_visit_mi_same[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_same'}, axis=1)

# Number of visits after analysis cutoff date
num_visits_after = pt_visit_mi_after[['person_id', 'visit_occurrence_id']].drop_duplicates().groupby('person_id').count().rename({'visit_occurrence_id' : 'num_visits_after'}, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC [Reference](https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns) for merging multiple dfs

# COMMAND ----------

# Merge number of visits before, during, and after analysis cutoff date to dataframe
num_visits_dfs = [mi_temp_8, num_visits_before, num_visits_same, num_visits_after]

mi_temp_9 = ft.reduce(lambda left, right: pd.merge(left, right, on='person_id', how='left'), num_visits_dfs).fillna(0)

# COMMAND ----------

mi_temp_9.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Obtain length in the EMR in months before, during, and after analysis cutoff date

# COMMAND ----------

# %%%% Before analysis cutoff date %%%%
# First visit start date
temp_1 = pt_visit_mi_before[['person_id', 'visit_start_date']].drop_duplicates().groupby('person_id').min().rename({'visit_start_date' : 'first_visit_start_date'}, axis=1)

# Last visit end date
temp_2 = pt_visit_mi_before[['person_id', 'visit_end_date']].drop_duplicates().groupby('person_id').max().rename({'visit_end_date' : 'last_visit_end_date'}, axis=1)

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
temp_1 = pt_visit_mi_after[['person_id', 'visit_start_date']].drop_duplicates().groupby('person_id').min().rename({'visit_start_date' : 'first_visit_start_date'}, axis=1)

# Last visit end date
temp_2 = pt_visit_mi_after[['person_id', 'visit_end_date']].drop_duplicates().groupby('person_id').max().rename({'visit_end_date' : 'last_visit_end_date'}, axis=1)

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
emr_months_dfs = [mi_temp_9, emr_months_before, emr_months_after]

mi_temp_10 = ft.reduce(lambda left, right: pd.merge(left, right, on='person_id', how='left'), emr_months_dfs).fillna(0)

# COMMAND ----------

mi_temp_10.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Add race, ethnicity, gender values and prune unneccessary columns

# COMMAND ----------

# Add concept_name for gender_concept_id
mi_temp_11 = mi_temp_10.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='gender_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'gender'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Add concept_name for race_concept_id
mi_temp_12 = mi_temp_11.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='race_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'race'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Add concept_name for ethnicity_concept_id
mi_temp_13 = mi_temp_12.merge(concepts[['concept_id', 'concept_name']], 
                              left_on='ethnicity_concept_id',
                              right_on='concept_id').rename({'concept_name' : 'ethnicity'}, 
                                                            axis=1).drop('concept_id', axis=1).copy()

# COMMAND ----------

# Retrieve columns
list(mi_temp_13.columns)

# COMMAND ----------

mi_temp_13['birth_datetime'].value_counts()

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
    #"location_source_value",
    #"adi",
    #"adi_category",
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

mi_temp_14 = mi_temp_13[cols_to_keep].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Check and save

# COMMAND ----------

# Check
print(f"Number of patients stay the same after refresh filtering: {mi_temp_10.shape[0] == mi_temp_8.shape[0]}")

# COMMAND ----------

# Save
mi_temp_14.to_csv("male_infertility_validation/demographics/mi_pts_only_final.csv")
mi_temp_14.to_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")
mi_temp_14.to_csv("male_infertility_validation/demographics/mi_pts_only_final.csv")
#mi_temp_14.write.mode("overwrite").parquet("male_infertility_validation/demographics/mi_pts_only_final.csv")


#a = pd.read_pickle("male_infertility_validation/demographics/mi_pts_only_final.pkl")
#a.to_csv("male_infertility_validation/demographics/mi_pts_only_final.csv")
## final 5551 with 19 variables

# COMMAND ----------

