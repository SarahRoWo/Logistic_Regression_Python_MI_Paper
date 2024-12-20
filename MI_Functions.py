# Databricks notebook source
# Updated 20241121

# COMMAND ----------

# Docstrings based on NumpPy docstring conventions
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

# COMMAND ----------

# install
!pip install matplotlib_venn
!pip install adjustText

# COMMAND ----------

# import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import mannwhitneyu 
from math import log10, log2
import math
import matplotlib_venn as mpv
import os
from adjustText import adjust_text

# COMMAND ----------

# format outputs
# ref: https://stackoverflow.com/questions/8924173/how-can-i-print-bold-text-in-python
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing functions for 01_Cohort Notebooks...**

# COMMAND ----------

# ADI category function
def ADI_category(adi_series_val):
  """
  Parameters
  __________
  adi_series_val : float
    mean ADI per person
  
  Returns
  _______
    ADI_category : string
      mean ADI category per person
  """
  
  if adi_series_val < 4:
    ADI_category = 'Low_ADI'
  elif (adi_series_val >= 4) and (adi_series_val < 7):
    ADI_category = 'Medium_ADI'
  elif (adi_series_val >= 7):
    ADI_category = 'High_ADI'
  else:
    print('check function and/or ADI values')

  return ADI_category

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing functions for 02_Diagnoses Notebooks...**

# COMMAND ----------

# References 

# Plotting multiple dataframes: https://stackoverflow.com/questions/22483588/how-to-plot-multiple-dataframes-in-subplots

# COMMAND ----------

def obtain_icd_snomed_diag(demo_df, pt_cond, concepts, file_name=None, save=False):
  """
  Parameters
  __________
  demo_df : pandas DataFrame 
    Contains demographic information of patients

  pt_cond : pandas DataFrame 
    Contains conditions of patients
  
  concepts : pandas DataFrame
    Contains concept names for conditions found in pt_cond
    
  file_name : string (default None)
    Denotes file name of the saved pandas DataFrame output
  
  save : bool (default False)
    Denotes whether to save results
    
  Returns
  _______
  first_diag_snomed_icd_final : pandas DataFrame 
    Contains icd and snomed diagnoses of patients
  """
  
  # Filter for patients of interest
  pt_cond_filtered = pt_cond[pt_cond['person_id'].isin(demo_df['person_id'])].copy()

  # Convert first_condition_start_date to datetime
  pt_cond_filtered['condition_start_date'] = pd.to_datetime(pt_cond_filtered['condition_start_date'], format='%Y-%m-%d')

  # Obtain first SNOMED diagnosis
  first_diag = pt_cond_filtered[['person_id', 'condition_concept_id', 'condition_source_concept_id', 'condition_start_date']].groupby(['person_id', 'condition_concept_id', 'condition_source_concept_id']).min().reset_index()

  display(first_diag)


  # Merge SNOMED concept_name, vocabulary_id, concept_class_id, domain_id, and concept_code
  temp_1 = first_diag.merge(concepts[['concept_id', 'concept_name', 'vocabulary_id', 'concept_class_id', 'domain_id', 'concept_code']], left_on='condition_concept_id', right_on='concept_id', how='left')
  first_diag_snomed = temp_1.rename({'concept_name' : 'snomed_concept_name',
                                    'vocabulary_id' : 'snomed_vocabulary_id',
                                    'concept_class_id' : 'snomed_concept_class_id',
                                    'domain_id' : 'snomed_domain_id',
                                    'concept_code' : 'snomed_concept_code'}, axis=1).drop('concept_id', axis=1)
  
  # Merge ICD concept_name, vocabulary_id, concept_class_id, domain_id, and concept_code
  temp_1 = first_diag_snomed.merge(concepts[['concept_id', 'concept_name', 'vocabulary_id', 'concept_class_id', 'domain_id', 'concept_code']], left_on='condition_source_concept_id', right_on='concept_id', how='left')
  temp_2 = temp_1.rename({'concept_name' : 'icd_concept_name',
                                        'vocabulary_id' : 'icd_vocabulary_id',
                                        'concept_class_id' : 'icd_concept_class_id',
                                        'domain_id' : 'icd_domain_id',
                                        'concept_code' : 'icd_concept_code'}, axis=1).drop('concept_id', axis=1)
  first_diag_snomed_icd = temp_2[(temp_2['icd_vocabulary_id'] == 'ICD9CM') | (temp_2['icd_vocabulary_id'] == 'ICD10CM')]

  # Add back patients
  first_diag_snomed_icd_final = demo_df['person_id'].to_frame().merge(first_diag_snomed_icd, on='person_id', how='left')
  
  # Check all patients now included
  print(f"All patients included: {first_diag_snomed_icd_final['person_id'].nunique() == demo_df['person_id'].nunique()}")
  
  # Save diagnoses
  if save:
    first_diag_snomed_icd_final_spark = spark.createDataFrame(first_diag_snomed_icd_final)
    first_diag_snomed_icd_final_spark.write.mode("overwrite").parquet("/mnt/ucsf-sarah-woldemariam-workspace/male_infertility/mi_logit_python/diagnoses/" + file_name)
    # Pandas
    #first_diag_snomed_icd_final.to_csv(f"path/to/{file_name}.csv")
  
  return first_diag_snomed_icd_final

  # COMMAND ----------

  def obtain_icd_snomed_diag_stanford(demo_df, pt_cond, concepts, file_name=None, save=False):
  """
  Parameters
  __________
  demo_df : pandas DataFrame 
    Contains demographic information of patients

  pt_cond : pandas DataFrame 
    Contains conditions of patients
  
  concepts : pandas DataFrame
    Contains concept names for conditions found in pt_cond
    
  file_name : string (default None)
    Denotes file name of the saved pandas DataFrame output
  
  save : bool (default False)
    Denotes whether to save results
    
  Returns
  _______
  first_diag_snomed_icd_final : pandas DataFrame 
    Contains icd and snomed diagnoses of patients
  """
  
  # Filter for patients of interest
  pt_cond_filtered = pt_cond[pt_cond['person_id'].isin(demo_df['person_id'])].copy()

  # Convert first_condition_start_date to datetime
  pt_cond_filtered['condition_start_date'] = pd.to_datetime(pt_cond_filtered['condition_start_date'], format='%Y-%m-%d')

  # Obtain first SNOMED diagnosis
  first_diag = pt_cond_filtered[['person_id', 'condition_concept_id', 'condition_source_concept_id', 'condition_start_date', 'condition_source_value']].groupby(['person_id', 'condition_concept_id', 'condition_source_concept_id', 'condition_source_value']).min().reset_index()

  first_diag = first_diag.rename({'condition_source_value' : 'icd'}, axis=1)

  display(first_diag)


  # Merge SNOMED concept_name, vocabulary_id, concept_class_id, domain_id, and concept_code; also merge condition_source_value
  temp_1 = first_diag.merge(concepts[['concept_id', 'concept_name', 'vocabulary_id', 'concept_class_id', 'domain_id', 'concept_code']], left_on='condition_concept_id', right_on='concept_id', how='left')
  first_diag_snomed = temp_1.rename({'concept_name' : 'snomed_concept_name',
                                    'vocabulary_id' : 'snomed_vocabulary_id',
                                    'concept_class_id' : 'snomed_concept_class_id',
                                    'domain_id' : 'snomed_domain_id',
                                    'concept_code' : 'snomed_concept_code'}, axis=1).drop('concept_id', axis=1)

  # Add back patients
  first_diag_snomed_icd_final = demo_df['person_id'].to_frame().merge(first_diag_snomed, on='person_id', how='left')
  
  # Check all patients now included
  print(f"All patients included: {first_diag_snomed_icd_final['person_id'].nunique() == demo_df['person_id'].nunique()}")
  
  # Save diagnoses
  if save:
    first_diag_snomed_icd_final_spark = spark.createDataFrame(first_diag_snomed_icd_final)
    first_diag_snomed_icd_final_spark.write.mode("overwrite").parquet(f"male_infertility_validation/diagnoses/{file_name}")
  
  return first_diag_snomed_icd_final

# COMMAND ----------

def check_icd_snomed_diag(diag_pts, demo_pts):
  """
  Parameters
  __________
  diag_pts : pandas DataFrame 
    Contains patients' icd and snomed diagnoses

  demo_pts : pandas DataFrame 
    Contains patients' demographic info. 
    
  NOTE : diag_pts and demo_pts should correspond to same patients
  
  Returns
  _______
  Nothing.
  """
  
  # Checking dimensions
  print(f"Dimensions of diag df is {diag_pts.shape}")
  
  # Exploring how many distinct diagnoses patients have
  temp_1 = diag_pts[['person_id', 'condition_source_concept_id']].groupby('person_id').nunique()
  temp_2 = temp_1.rename({'condition_source_concept_id' : 'num_icd_diagnoses'}, axis=1)
  num_diag_per_pt = temp_2.sort_values(by='num_icd_diagnoses', ascending=False)

  print("Number of distinct ICD diagnoses per patient")
  display(num_diag_per_pt)
  
  
  # Check that the only ICD_vocabulary_ids are ICD9CM and ICD10CM
  icd_vocab_check = diag_pts['icd_vocabulary_id'].unique()
  print("Checking that the only ICD_vocabulary_ids that show up are ICD9CM and ICD10CM")
  display(icd_vocab_check)
  
  # Check that all patients are accounted for
  num_pts_demo = demo_pts['person_id'].nunique()
  num_pts_diag = diag_pts['person_id'].nunique()
  
  print("\n\nChecking that the same number of patients are in the demo and diag dfs")
  print("Number of patients in demographics df")
  display(num_pts_demo)
  
  print('\n')
  
  print("Number of patients in diagnoses df")
  display(num_pts_diag)

# COMMAND ----------

def obtain_phecodes(icd_diag, phecodes, file_name=None, save=False):
  """
  Parameters
  __________
  icd_diag : pandas DataFrame 
    Contains patients' icd and snomed diagnoses
    Denotes file name of the saved pandas DataFrame output
  phecodes : spark DataFrame
    Contains phecodes with original excl_phenotypes converted to icd10-like chapters when applicable.
    Merge of phecoeds_v2 and phecodes_cat_v2
  file_name : string (default None)
    Name of file if saving
  save : bool (default False)
    Whether to save results
  
  Returns
  _______
  phe_diag : spark DataFrame 
    Contains patient's icd, snomed, and corresponding phecode diagnoses
  """

  # convert icd_diag to spark DataFrame
  icd_diag = spark.createDataFrame(icd_diag)

  temp_1 = icd_diag.join(phecodes, 
                         icd_diag.icd_concept_code == phecodes.icd,
                         'left')
  temp_2 = temp_1.dropDuplicates()
  phe_diag = temp_2.drop('icd')
  
  if save:
    print('Saving...')
    phe_diag.write.mode("overwrite").parquet("/mnt/ucsf-sarah-woldemariam-workspace/male_infertility/mi_logit_python/diagnoses/" + file_name)
    print('Saved')

  return phe_diag

# COMMAND ----------

def obtain_phecodes_stanford(icd_diag, phecodes, file_name=None, save=False):
  """
  Parameters
  __________
  icd_diag : pandas DataFrame 
    Contains patients' icd and snomed diagnoses
    Denotes file name of the saved pandas DataFrame output
  phecodes : pandas DataFrame
    Contains phecodes with original excl_phenotypes converted to icd10-like chapters when applicable.
    Merge of phecoeds_v2 and phecodes_cat_v2
  file_name : string (default None)
    Name of file if saving
  save : bool (default False)
    Whether to save results
  
  Returns
  _______
  phe_diag : pandas DataFrame 
    Contains patient's icd, snomed, and corresponding phecode diagnoses
  """

  temp_1 = icd_diag.merge(phecodes, left_on='condition_source_value', right_on='icd', how='left').copy()
  temp_2 = temp_1.drop_duplicates().copy()
  phe_diag = temp_2.drop('condition_source_value', axis=1).copy()
  
  if save:
    print('Saving...')
    phe_diag.to_csv(f"male_infertility_validation/diagnoses/{file_name}.csv")
    print('Saved')

  return phe_diag

def add_first_phe_date(df):
  """
  Adds info re: first phenotype start date 
  since multiple conditions can aggregate into one phenotype

  Parameters
  __________
  df : pandas DataFrame
    Contains diagnoses information, including phecodes
  
  Returns
  _______
  first_phe_date : pandas dataframe
    Contains first date of phenotype for each patient
  """

  # Get first phenotype date per person
  df = (
      df[['phenotype', 'person_id', 'condition_start_date']].groupby(['person_id', 'phenotype']).min()
  )

  df = df.reset_index()

  df = df.rename({'condition_start_date' : 'first_phe_start_date'}, axis=1)

  first_phe_date = df.copy()

  return first_phe_date

# COMMAND ----------

def add_phe_rel_date(df, first_phe_date, cutoff_date):
  """
  Adds columns to diagnoses df specifying whether first phenotype start date
  for a given phenotype occurs before or after first mi diagnosis
  or first vasectomy procedure

  Parameters
  __________
  df : pandas DataFrame
    Contains diagnoses information, including phecodes

  first_phe_date : pandas DataFrame
    Contains first date of phenotype for each patient
  
  cutoff_date : string
    Date of cutoff 6 months after first male infertility diagnosis - 'mi_analysis_cutoff_date'
    Date of cutoff 6 months after first vasectomy procedure - 'vas_analysis_cutoff_date'

  Returns
  _______
  combined : pandas DataFrame
    Contains diagnoses information, plus:
    - first phenotype date for a given patient
    - whether it occurred before, after, or at the same time
    as mi diagnosis or procedure

  """

  # Join first_phe_start_date
  df = df.merge(first_phe_date, how='left', left_on=['phenotype', 'person_id'], right_on=['phenotype', 'person_id'])

  # Make new rows determining whether first_phe_start_date occurred before or after mi diagnosis
  df['phe_time_before'] = df['first_phe_start_date'] < df[cutoff_date]
  df['phe_time_after'] = df['first_phe_start_date'] > df[cutoff_date]
  df['phe_time_same'] = df['first_phe_start_date'] == df[cutoff_date]

  combined = df.copy()

  return combined

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing functions for 03_Logistic_Regression Notebooks...**

# COMMAND ----------

# MAGIC %md
# MAGIC *Importing non plotting functions for 03_Logistic_Regression Notebooks...*

# COMMAND ----------

def filter_by_cutoff(demo, phe, cutoff_time, phe_analysis_cutoff='after'):
    """
    Parameters
    __________
    demo : pandas DataFrame
        Contains demographic information of patients
    
    phe : pandas DataFrame
        Contains diagnosis information of patients, represented as phenotypes

    cutoff_time : int or float
        Number of months used to define cutoff time
    
    phe_analysis_cutoff : string (default 'after')
        Whether we are looking at diagnoses before or after cutoff date. Default 'after'. 
        Alternative values are 'before' and 'same'

    Returns
    _______
    phe_cutoff_final : pandas DataFrame
        Contains diagnosis information of patients, represented as phenotypes, within a given cutoff time
    """

    print(f"Cutoff time is {cutoff_time} months")
    demo_cutoff = demo[demo['emr_months_after'] >= cutoff_time].copy()

    # Patients and their phenotypes after meeting cutoff
    phe_cutoff = phe[phe['person_id'].isin(demo_cutoff['person_id'])].copy()
    print(f"Number of unique patients after cutoff: {phe_cutoff['person_id'].nunique()}")

    # Only look at patient phenotypes before or after analysis cutoff date
    phe_cutoff2 = phe_cutoff[phe_cutoff['phe_time_'+phe_analysis_cutoff] == 1].copy()

    # Obtain diagnoses within cutoff time for patients with at least cutoff_time amount of followu
    phe_cutoff_final = phe_cutoff2[phe_cutoff2['diag_cutoff_delta_approx_m'] <= abs(cutoff_time)].copy()

    return demo_cutoff, phe_cutoff_final

# COMMAND ----------

def combine_diagnoses(df_case, df_con):
  """
  Combine diagnoses of cases and controls

  Parameters
  __________
  df_case : pandas DataFrame
    Contains diagnoses information (including phecode-corresponding phenotypes)
    for cases (e.g. patients with paternal infertility)

  df_con : pandas DataFrame
    Contains diagnoses information (including phecode-corresponding phenotypes)
    for control patients

  Returns
  _______
  diag_combined : pandas DataFrame
    Contains phecodes, phenotypes, corresponding phecode_category information
    for each patient; also includes information on whether they are case or
    control
  """
  
  df_case = df_case[['person_id', 'phenotype', 'phecode']].drop_duplicates()
  df_case['has_mi'] = 1

  df_con = df_con[['person_id', 'phenotype', 'phecode']].drop_duplicates()
  df_con['has_mi'] = 0

  diag_combined = pd.concat([df_case, df_con])

  # remove diagnoses where there is no corresponding phenotype
  diag_combined = diag_combined[~diag_combined['phenotype'].isnull()]

  return diag_combined

# COMMAND ----------

def combine_demographics(mi_demo, con_demo):
  """
  Parameters
  __________
  mi_demo : pandas DataFrame
    Contains demographics info for patients with paternal infertility
    
  con_demo : pandas DataFrame
    Contains demographics info for control patients 

  Returns
  _______
  demo_combined : pandas DataFrame
    Contains demographics info for all patients, including their pi_status
  """

  # Specify whether patient has pi
  mi_demo['has_mi'] = 1
  con_demo['has_mi'] = 0

  demo_combined = pd.concat([mi_demo, con_demo])

  return demo_combined

# COMMAND ----------

def Xy_logreg(diag_combined, demo_combined, phenotype):
  """
  Prepare Xy df for a given phenotype

  Parameters
  __________
  diag_combined : pandas DataFrame
    Contains person_id, phecodes, phenotypes, phecode_category, and whether patient has male infertility
    
  phenotype : string
    Phenotype of interest

  Returns
  _______
  Xy_final : pandas Dataframe
    Contains whether patient has been diagnosed with phenotype of interest as well as associated demographics for each patient
  """

  # select rows for phenotype of interest
  df_comb_phe = diag_combined[diag_combined['phenotype'] == phenotype]

  # obtain person_ids for those who have the phenotype
  pts_w_phenotype = df_comb_phe[['person_id', 'has_mi']]
  pts_w_phenotype.insert(loc=1, column='has_phenotype', value=1)

  # obtain person_ids for those who don't have the phenotype
  demo_combined_temp = demo_combined[['person_id', 'has_mi']]
  pts_wo_phenotype = demo_combined_temp[~demo_combined_temp['person_id'].isin(pts_w_phenotype['person_id'])]
  pts_wo_phenotype.insert(loc=1, column='has_phenotype', value=0)

  # concat 
  Xy = pd.concat([pts_w_phenotype, pts_wo_phenotype])

  # add demographic information
  Xy_final = Xy[['person_id', 'has_phenotype']].merge(demo_combined, on='person_id').drop_duplicates()
  
  return Xy_final

# COMMAND ----------

def Xy_clean(Xy_pre):
  """
  Changes null values to 0 and converts dtype object into category
  Parameters
  __________
  Xy_pre : pandas DataFrame
    Contains covariates, including whether patient has phenotype / male infertility

  Returns
  _______
  Xy_clean : pandas DataFrame
    Converts object dtype to category
  """

  # fill na with 0 for num_visits_before, num_visits_after, num_visits_same
  Xy_pre['num_visits_before'] = Xy_pre['num_visits_before'].fillna(0)
  Xy_pre['num_visits_after'] = Xy_pre['num_visits_after'].fillna(0)
  Xy_pre['num_visits_same'] = Xy_pre['num_visits_same'].fillna(0)

  # convert object dtypes into category

  # columns where dtype is object
  object_dtypes = Xy_pre.select_dtypes(include='object').columns

  # convert those columns into category dtype
  Xy_pre[object_dtypes] = Xy_pre[object_dtypes].astype('category')

  Xy_clean = Xy_pre

  return Xy_clean

# COMMAND ----------

def run_logistic_regresssion(diag_combined, demo_combined, formula):
  """
  Runs logistic regression for each phenotype

  Parameters
  __________
  diag_combined : pandas DataFrame
    Contains all patients' diagnoses
    Note this function assumes that at least 1 male infertility and 1 control patient
    has a given phenotype

  demo_combined : pandas DataFrame
    Contains all patients' demographics information
    
  formula : string
    Specific analysis to be performed.
    For example, formula for crude model would be:
    formula='has_mi ~ has_phenotype'

  Returns
  _______
  results_all : pandas DataFrame
    Contains results for logistic regression models for all phenotypes
  
  """
  results = list()
  sing_mat_phe = list()

  print(f"Number of phenotypes explored: {diag_combined['phenotype'].nunique()}")

  for phenotype in tqdm(diag_combined['phenotype'].unique()):

    # prepare Xy for given phenotype
    
    try:
      Xy_pre = Xy_logreg(diag_combined=diag_combined, demo_combined=demo_combined, phenotype=phenotype)
      Xy = Xy_clean(Xy_pre)
      #Xy['has_phenotype'] = Xy['has_phenotype']+0.00001*np.random.rand(Xy_pre2.shape[0])

      # run logistic regression 
      mod = smf.logit(formula=formula, data=Xy)
      res = mod.fit(method='newton', maxiter=500, disp=False)

      # obtain summary of results
      summary = res.summary()
      
      # obtain summary of results as a DataFrame row
      # first, save as html
      results_as_html = summary.tables[1].as_html()
      summary_pd = pd.read_html(results_as_html, header=0, index_col=0)[0]

      # second, flatten DataFrame to one row
      summary_pd = summary_pd.unstack().to_frame().sort_index(level=1).T
      summary_pd.columns = summary_pd.columns.map('_'.join)

      # insert phenotype column and phenotype name into results DataFrame
      summary_pd.insert(0, column='phenotype', value=[phenotype])

      # add odds ratio for phenotype
      summary_pd['odds_ratio_has_phenotype'] = summary_pd['coef_has_phenotype'].apply(lambda x: np.exp(x))  

      # add whether model converged
      summary_pd['converged'] = res.mle_retvals['converged']

      # drop p-value column for phenotype
      summary_pd = summary_pd.drop('P>|z|_has_phenotype', axis=1)

      # add back more precise p-value for phenotype
      summary_pd['P>|z|_has_phenotype'] = res.pvalues['has_phenotype']

      results.append(summary_pd)            
  except np.linalg.LinAlgError as err:
    print(f"Oops! There was a linear algebra error: {err}")
    print(phenotype)
    sing_mat_phe.append(phenotype)

  print(f"Number of phenotypes with singluar matrix: {len(sing_mat_phe)}")
  
  # concatenate DataFrame rows for DataFrame containing logistic regression
  # results for each phenotype
  results_all = pd.concat([x for x in results])

  return results_all

# COMMAND ----------

def run_logistic_regression_rvrs(diag_combined, demo_combined, formula):
  """
  Runs logistic regression for each phenotype as a predictor variable

  Parameters
  __________
  diag_combined : pandas DataFrame
    Contains all patients' diagnoses
    Note this function assumes that at least 1 male infertility and 1 control patient
    has a given phenotype

  demo_combined : pandas DataFrame
    Contains all patients' demographics information
    
  formula : string
    Specific analysis to be performed.
    For example, formula for crude model would be:
    formula='has_mi ~ has_mi'

  Returns
  _______
  results_all : pandas DataFrame
    Contains results for logistic regression models for all phenotypes
  
  """
  results = list()
  sing_mat_phe = list()

  print(f"Number of phenotypes explored: {diag_combined['phenotype'].nunique()}")

  for phenotype in tqdm(diag_combined['phenotype'].unique()):

    try:

      # prepare Xy for given phenotype
      Xy_pre = Xy_logreg(diag_combined=diag_combined, demo_combined=demo_combined, phenotype=phenotype)
      Xy = Xy_clean(Xy_pre)
      #Xy['has_mi'] = Xy['has_mi']+0.00001*np.random.rand(Xy_pre2.shape[0])

      # run logistic regression 
      mod = smf.logit(formula=formula, data=Xy)
      res = mod.fit(method='newton', maxiter=500, disp=False)

      # obtain summary of results
      summary = res.summary()
      
      # obtain summary of results as a DataFrame row
      # first, save as html
      results_as_html = summary.tables[1].as_html()
      summary_pd = pd.read_html(results_as_html, header=0, index_col=0)[0]

      # second, flatten DataFrame to one row
      summary_pd = summary_pd.unstack().to_frame().sort_index(level=1).T
      summary_pd.columns = summary_pd.columns.map('_'.join)

      # insert phenotype column and phenotype name into results DataFrame
      summary_pd.insert(0, column='phenotype', value=[phenotype])

      # add odds ratio for phenotype
      summary_pd['odds_ratio_has_mi'] = summary_pd['coef_has_mi'].apply(lambda x: np.exp(x))  

      # add whether model converged
      summary_pd['converged'] = res.mle_retvals['converged']

      # drop p-value column for phenotype
      summary_pd = summary_pd.drop('P>|z|_has_mi', axis=1)

      # add back more precise p-value for phenotype
      summary_pd['P>|z|_has_mi'] = res.pvalues['has_mi']

      results.append(summary_pd)

      if phenotype == 'Kyphoscoliosis and scoliosis':
        print('log reg completed removing one UCSD patient')
    except np.linalg.LinAlgError as err:
      print(f"Oops! There was a linear algebra error: {err}")
      print(phenotype)
      sing_mat_phe.append(phenotype)
      #corr_matrix = pd.get_dummies(Xy[['has_phenotype', 'has_mi', 'location_source_value', 'mi_or_vas_est_age']]).#corr()
      #display(corr_matrix) 
      #eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
      #print(f"Eigenvalues are {eigenvalues} and eigenvectors are {eigenvectors}")
      #break

  
  print(f"Number of phenotypes with singular matrix: {len(sing_mat_phe)}")
  print(f"Phenotypes with singular matrix: {sing_mat_phe}")
  # concatenate DataFrame rows for DataFrame containing logistic regression
  # results for each phenotype
  results_all = pd.concat([x for x in results])

  return results_all

# COMMAND ----------

def clean_col_names(results_df):
  """
  Cleans up column names for readability

  Parameters
  __________
  results_df : pandas DataFrame
    Contains logistic regression results for each phenotype

  Returns
  _______
  results_clean_df : pandas DataFrame
    Same as result_df, except with more readable column names  
  """

  # change p-value column names to start with pval
  pval_cols = results_df.filter(regex='P>').copy()
  
  for column in pval_cols.columns:
    results_df = results_df.rename({column : 'pval' + column[5:]}, axis=1)

  # change 95% confidence interval lower bound column names to start with ci_lb
  ci_lb_cols = results_df.filter(regex='0.025')

  for column in ci_lb_cols.columns:
    results_df = results_df.rename({column : 'ci_lb' + column[6:]}, axis=1)

  # change 95% confidence interval upper bound column names to start with ci_ub
  ci_ub_cols = results_df.filter(regex='0.975')

  for column in ci_ub_cols.columns:  
    results_df = results_df.rename({column : 'ci_ub' + column[6:]}, axis=1)
  
  results_clean_df = results_df.copy()

  return results_clean_df

# COMMAND ----------

def significance_bc(results_df):
  """
  Adds bonferroni-corrected significance column. Specifies whether:
  1. phenotype is significantly associated for patients with male infertility
  2. phenotype is signficiantly associated for control patients
  3. phenotype is not significant
  
  Parameters
  __________
  results_df : pandas DataFrame
    Contains results for logistic regression models for each phenotype
  
  Returns
  _______
  results_df_sig : pandas DataFrame
    Contains results for logistic regression models for each phenotype and
    its significance

  """
  # Bonferonni corrected p-value
  bc = .05 / results_df.shape[0] 

  print('bc:',bc)

  # obtain significance for each phenotype
  sig = np.full(shape=(results_df.shape[0],), fill_value='not_significant') 
  mask = (results_df['pval_has_phenotype'] < bc) & (results_df['odds_ratio_has_phenotype'] > 1)
  sig[mask] = 'mi_significant'
  mask = (results_df['pval_has_phenotype'] < bc) & (results_df['odds_ratio_has_phenotype'] < 1)
  sig[mask] = 'con_significant'

  results_df['significance'] = sig

  results_df_sig = results_df.copy()

  return results_df_sig

# COMMAND ----------

def significance_bc_rvrs(results_df):
  """
  Adds bonferroni-corrected significance column. Specifies whether:
  1. male infertility is significantly associated with phenotype
  2. vasectomy is signficiantly associated with phenotype
  3. having male infertility or vasectomy is not significant
  
  Parameters
  __________
  results_df : pandas DataFrame
    Contains results for logistic regression models for each phenotype
  
  Returns
  _______
  results_df_sig : pandas DataFrame
    Contains results for logistic regression models for each phenotype and
    its significance

  """
  # Bonferonni corrected p-value
  bc = .05 / results_df.shape[0] 

  print('bc:',bc)

  # obtain significance for each phenotype as it related to having male infertility
  sig = np.full(shape=(results_df.shape[0],), fill_value='not_significant') 
  mask = (results_df['pval_has_mi'] < bc) & (results_df['odds_ratio_has_mi'] > 1)
  sig[mask] = 'mi_significant'
  mask = (results_df['pval_has_mi'] < bc) & (results_df['odds_ratio_has_mi'] < 1)
  sig[mask] = 'con_significant'

  results_df['significance'] = sig

  results_df_sig = results_df.copy()

  return results_df_sig

# COMMAND ----------

def significance_bh(logit_model_df):
  """
  Adds significance from BH-adjusted p-value

  Parameters
  __________
  logit_model_df : pandas DataFrame
    Contains logistic regression model results for each phenotype

  Returns 
  _______
  significance_bh : string
    Whether phenotype is significant for patients with male infertility,
    control patients, or neither
  """

  if (logit_model_df['odds_ratio_has_phenotype'] > 1) and (logit_model_df['pval_BH_adj_sig'] == True):
    significance_bh =  'mi_significant'
    return significance_bh
  elif (logit_model_df['odds_ratio_has_phenotype'] < 1) and (logit_model_df['pval_BH_adj_sig'] == True):
    significance_bh =  'con_significant'
  else:
    significance_bh = 'not_significant'

  return significance_bh

# COMMAND ----------

def significance_bh_rvrs(logit_model_df):
  """
  Adds significance from BH-adjusted p-value

  Parameters
  __________
  logit_model_df : pandas DataFrame
    Contains logistic regression model results for each phenotype

  Returns 
  _______
  significance_bh : string
    Whether phenotype is significant for patients with male infertility,
    control patients, or neither
  """

  if (logit_model_df['odds_ratio_has_mi'] > 1) and (logit_model_df['pval_BH_adj_sig'] == True):
    significance_bh =  'mi_significant'
    return significance_bh
  elif (logit_model_df['odds_ratio_has_mi'] < 1) and (logit_model_df['pval_BH_adj_sig'] == True):
    significance_bh =  'con_significant'
  else:
    significance_bh = 'not_significant'

  return significance_bh

# COMMAND ----------

# Modified from Alice Tang
def countPtsDiagnosis_Dict_LR(diag_df, totalpts, diagkeys):
    """
    Count number of patients with male infertilit and vasectomy with a given phenotype
    diagnosisdict = countPtsWithWithoutDiagnosis(csvfile, totalpts)

    This doesn't include phecode_category
    
    Parameters
    __________
    diag_df : pandas DataFrame
      Contains patients' diagnoses
      Columns should include person_id, phenotype, and diagkeys
    totalpts : int 
      Represents total number of patients in data
    diagkeys : list 
      Contains diagnostic categories represented as strings. 
    
    Returns
    _______
    diagnosisdict : dictionary 
      Diagnosis dictionary of pandas DataFrames. 
      
      diagnosisdict[x] will give you the DataFrame for a diagkey category.
                       Each DataFrame includes the following columns: diagkey_category, Count, Count_r, where Count is the 
                       number of patients with a diagnosis, and Count_r = #totalpts - Count                    
    """
       
    # Loop through diagkeys and save counts 
    diag_dfCount = dict()
    for n in diagkeys:
        diagtemp = diag_df[['person_id',n]].drop_duplicates() # drop duplicate diagnosis for each patient
        
        diag_dfCount[n]= pd.DataFrame(diagtemp[n].value_counts()).reset_index()
        diag_dfCount[n].columns = [n,'Count']
        diag_dfCount[n]['Count_r'] = totalpts - diag_dfCount[n]['Count']
        
    return diag_dfCount

# COMMAND ----------

# Modified from Alice Tang
def countPtsDiagnosis_Dict_LR_rvrs(diag_df, totalpts, diagkeys):
    """
    diagnosisdict = countPtsWithWithoutDiagnosis(csvfile, totalpts)

    This doesn't include phecode_category
    
    Parameters
    __________
    diag_df : pandas DataFrame
      Contains patients' diagnoses
      Columns should include person_id, phenotype, and diagkeys
    totalpts : int 
      Represents total number of patients in data
    diagkeys : list 
      Contains diagnostic categories represented as strings. 
    
    Returns
    _______
    diagnosisdict : dictionary 
      Diagnosis dictionary of pandas DataFrames. 
      
      diagnosisdict[x] will give you the DataFrame for a diagkey category.
                       Each DataFrame includes the following columns: diagkey_category, Count, Count_r, where Count is the 
                       number of patients with a diagnosis, and Count_r = #totalpts - Count                    
    """
       
    # Loop through diagkeys and save counts 
    diag_dfCount = dict()
    for n in diagkeys:
        diagtemp = diag_df[['person_id',n]].drop_duplicates() # drop duplicate diagnosis for each patient
        
        diag_dfCount[n]= pd.DataFrame(diagtemp[n].value_counts()).reset_index()
        diag_dfCount[n].columns = [n,'Count']
        diag_dfCount[n]['Count_r'] = totalpts - diag_dfCount[n]['Count']
        
    return diag_dfCount

# COMMAND ----------

# Modified from Alice
def countPtsDiagnosis_Dict(csvfileordf, totalpts, diagkeys):
    """
    diagnosisdict = countPtsWithWithoutDiagnosis(csvfile, totalpts)
    
    Parameters
    __________
    csvfileordf : string OR pandas DataFrame
      1. if string, denotes csv file path containing patients' diagnoses 
      2. if pandas DataFrame, denotes pandas DataFrame of patients' diagnoses
      In both cases, columns of csv/df should include person_id, phenotype, and diagkeys
    totalpts : int 
      Represents total number of patients in data
    diagkeys : list 
      Contains diagnostic categories represented as strings. 
    
    Returns
    _______
    diagnosisdict : dictionary 
      Diagnosis dictionary of pandas DataFrames. 
      
      diagnosisdict[x] will give you the DataFrame for a diagkey category.
                       Each DataFrame includes the following columns: diagkey_category, Count, Count_r, where Count is the 
                       number of patients with a diagnosis, and Count_r = #totalpts - Count                    
    """
    
    # If csv file:
    if type(csvfileordf)==str: 
        # Read in file
        ptDiag = pd.read_csv(csvfileordf) 
        if not(ptDiag['person_id'].unique().shape[0] == totalpts):
            raise Exception('Patient Cohort Unique ID number doesn''t match up to Pt Number')
    # If dataframe:
    elif type(csvfileordf)==pd.DataFrame: 
        ptDiag = csvfileordf
    else:
        raise Exception(str(type(csvfileordf)) + 'not supported. Please give csv file name as string, or dataframe.')
       
    # Loop through diagkeys and save counts 
    ptDiagCount = dict()
    for n in diagkeys:
        diagtemp = ptDiag[['person_id',n]].drop_duplicates() # drop duplicate diagnosis for each patient
        
        ptDiagCount[n]= pd.DataFrame(diagtemp[n].value_counts()).reset_index()
        ptDiagCount[n].columns = [n,'Count']
        ptDiagCount[n]['Count_r'] = totalpts - ptDiagCount[n]['Count']
        
    return ptDiagCount

# COMMAND ----------

# For making log-log plots
def sig_bh_overlap_det(df, comp, primary='primary'):
  """
  Parameters
  __________

  df : pandas DataFrame
    Contains significance and log10(OR) for each phenotype for primary analysis and comparison analysis

  comp : str
    Comparison analysis in df

  primary : str (default 'primary')
    The 'primary' analysis being compared to

  Returns
  _______
  sig_overlap : str
    Whether phenotype is significant in both analyses, one analysis, or neither
  """
  
  if (df['significance_bh_' + primary] == 'mi_significant') and (df['significance_bh_' + comp] == 'mi_significant'):
    sig_overlap = 'mi_sig_both'
  elif (df['significance_bh_' + primary] == 'mi_significant') and (df['significance_bh_' + comp] != 'mi_significant'):
    sig_overlap = 'mi_sig_' + primary
  elif (df['significance_bh_' + primary] != 'mi_significant') and (df['significance_bh_' + comp] == 'mi_significant'):
    sig_overlap = 'mi_sig_' + comp
  elif (df['significance_bh_' + primary] == 'con_significant') and (df['significance_bh_' + comp] == 'con_significant'):
    sig_overlap = 'con_sig_both'
  elif (df['significance_bh_' + primary] == 'con_significant') and (df['significance_bh_' + comp] != 'con_significant'):
    sig_overlap = 'con_sig_' + primary
  elif (df['significance_bh_' + primary] != 'con_significant') and (df['significance_bh_' + comp] == 'con_significant'):
    sig_overlap = 'con_sig_' + comp
  else:
    sig_overlap = 'not_significant'

  return sig_overlap

# COMMAND ----------

# For making upset plots
def overall_significance_mi(df, suffixes=['_primary', '_sdoh', '_hosp']):
    """
    Get overlapping significances for patients with male infertility
    
    Parameters
    __________
    df : pandas DataFrame
      Contains significances of phenotype per logistic regression model
      Contains the following columns:
      1. 'sig_primary'
      2. 'sig_sdoh'
      3. 'sig_hosp'

    suffixes : list (default ['_primary', '_sdoh', '_hosp'])
      Suffixes of each analysis
    
    Returns
    _______
    significance : tuples
      contains which models phenotype is significant for 
    """

    significance = list()
    suffixes = suffixes
    for suffix in suffixes:
        if df['sig'+suffix] == 'mi_significant':      
            significance.append('significance_bh'+suffix)
    significance = tuple(significance)
    return significance

# COMMAND ----------

# For making upset plots
def overall_significance_con(df, suffixes=['_primary', '_sdoh', '_hosp']):
    """
    Get overlapping significances for control patients
    
    Parameters
    __________
    df : pandas DataFrame
      Contains significances of phenotype per logistic regression model
      Contains the following columns:
      1. 'sig_primary'
      2. 'sig_sdoh'
      3. 'sig_hosp'

    suffixes : list (default ['_primary', '_sdoh', '_hosp'])
      Suffixes of each analysis
    
    Returns
    _______
    significance : tuples
      contains which models phenotype is significant for 
    """

    significance = list()
    suffixes = suffixes
    for suffix in suffixes:
        if df['sig'+suffix] == 'con_significant':      
            significance.append('significance_bh'+suffix)
    significance = tuple(significance)
    return significance

# COMMAND ----------

# For making upset plots
def abv(sig_column):
  """
  Creates list of overlapping significance for each phenotype

  Parameters
  __________
  sig_column : Pandas series
    Column contaning overlapping significance in a tuple

  Returns
  _______
  sig_upset : list
    Contains overlapping significance for each phenotype as a list
    Abbreviates the significance so that only the analysis name is included
  """

  sig_upset = list()

  for match_strategy in sig_column:
    sig_upset.append(match_strategy[16:])

  return sig_upset

# COMMAND ----------

# For censoring results
def set_to_ten(count):
    """
    Parameters
    __________
    count : int
      Patient count 

    Returns
    _______
    count : int 
      If applicable, censored patient count that is set to 10. 
      If not applicable, returns original patient count
    """
  
    if count <= 10:
        count = 10
        
    return count

# COMMAND ----------

# MAGIC %md
# MAGIC *Importing plotting functions for 03_Logistic_Regression Notebooks...*

# COMMAND ----------

def volcano(analysis, 
            file_name_analysis, 
            hue_order=['male infertility significant', 'vasectomy significant', 'not significant'],
            s=12, 
            save=0, 
            file_name_save=None, 
            figtype='pdf', 
            dims=(8,6), 
            legend_loc=(1.02, 0.88),
            textcolor='k',
            annotate=False,
            annotatefontsize=12):
  """
  Parameters
  __________
  analysis : pandas DataFrame
    Contains enrichment analyses results

  file_name_analysis : string
    Specifies the specific enrichment analysis associated with analysis parameter

  hue_order : list (default ['male infertility significant', 'vasectomy significant', 'not significant'])
    Contains order of significance
    Elements are strings

  s : int (default 12)
    Specifies size of marker

  save : bool (default 0)
    Whether to save plot 

  file_name_save : string (default None)
    Where to save plot, if save parameter is set to 1 (True)

  figtype : string
    Specifies file type of plot, if save parameter is set to 1 (True)

  dims : tuple : (default (8,6))
    Specifies dimensions of plot in 2-element tuple usually comprised of ints

  legend_loc : tuple (default (1.02, 0.88))
    Specifies location of legend
    
  textcolor : string (default 'k')
    Color of annotated text

  annotate : bool (default False)
    Whether to annotate top phenotypes

  annotatefontsize : int (default 12)
    Font size of annotated text (if annotate set to True)
  
  Returns
  _______
  Nothing.
  """
  print(color.BOLD + 'Volcano plot for ' + file_name_analysis + color.END)
  
  analysis_pd = analysis

  # change significance column values
  analysis_pd['significance_bh'] = analysis_pd['significance_bh'].replace({'mi_significant' : 'male infertility significant',
                                                   'con_significant' : 'vasectomy significant',
                                                   'not_significant' : 'not significant'})
  
  
  bc = .05/analysis_pd.shape[0]

  fig = plt.figure(figsize=dims)
  g = sns.scatterplot(data=analysis_pd, 
                      x='log10_OR_has_phenotype', 
                      y='-log10_pval_has_phenotype', 
                      hue='significance_bh',
                      hue_order=hue_order,
                      edgecolor=None,
                      palette=['hotpink', 'dodgerblue', 'lightgrey'],
                      s=s)
  
  texts = []
  kwargs = dict(color=textcolor, fontweight='medium')
  #kwargs2 = dict(color='k', fontweight='heavy')
  
  # Add annotation (20220111)
  # source: https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas-dataframe/39374693
  if annotate:
    for idx, row in analysis_pd.iterrows():
      if row['annotate'] == 1:
          texts.append(g.annotate(row['phenotype'], (row['log10_OR_has_phenotype'], row['-log10_pval_has_phenotype']), fontsize=annotatefontsize, **kwargs))
          
  if len(texts) > 0:
    adjust_text(texts, arrowprops=dict(arrowstyle="-", 
                                       color='k')) # #A4C2EA
    # arrowprops=dict(arrowstyle='-|>, head_width=.1', color="k", shrinkA=3, shrinkB=3
  
  # determine x-axis min and max values
  min_x = np.abs(analysis_pd['log10_OR_has_phenotype'].min())
  max_x = np.abs(analysis_pd['log10_OR_has_phenotype'].max())

  if min_x > max_x:
    plt.xlim([-1*min_x, min_x])
  else:
    plt.xlim([-1*max_x, max_x])
  
  plt.axvline(0, color='#555555',linestyle=':')
  plt.xlabel(r'$\log_{10}$(Odds Ratio)', fontsize=24)
  plt.xticks(fontsize=20)
  plt.yticks(fontsize=20)
  plt.ylabel(r'$-\log_{10}$(p-value)', fontsize=24)    
  plt.legend(loc=legend_loc, fontsize=18)
  plt.title('Male Infertility vs. Vasectomy \nPhenotype Volcano Plot\n', fontsize=26, fontweight='bold')

  if save:
    print('Saving...')
    plt.savefig(f"/dbfs/FileStore/{file_name_save}.pdf", bbox_inches='tight')
    print('Saved')

  print('\n')

  plt.show()

# COMMAND ----------

def mhat_no_line_breaks(df, # pandas DataFrame
                        groupby_col=None, # column to group by
                        logp=None, # df column containing -log 10 of the p-value
                        
                        # Figure parameters
                        dim=(20,10), 
                        
                        sig_pval_line=False, # bool that adds significance cutoff line
                        sig_pval=0.05, # significant p-value cutoff
                        
                        axxlabel=None, # x-axis label for bottom subplot 
                        axylabel=None, # y-axis label (used?)
                        axlabelfontsize=9, # x-axis label fontsize for bottom subplot
                        axlabelfontname="DejaVu Sans", # font used
                        axtickfontsize=9, # x-axis tick label fontsize
                        axtickfontname="DejaVu Sans", # font for xticks 
                        ar=90, # How much to rotate phecode category name on x axis
                        
                        dotsize=8, # size of markers
                        figtitle='Manhattan Plot', # figure title
                        
                        # Annotation parameters
                        annotate=True, # Whether to annotate
                        annotatefontsize=20, # annotation font size
                        expand_text=(1.05, 1.2), 
                        expand_points=(1.05, 1.2),
                        autoalign=True, 
                        
                        # Save parameters
                        save=False,
                        filename='Manhattan_Plot.pdf'): 

  """
  Makes Manhattan plot of data

  Parameters
  __________
  df : pandas DataFrame
    Contains data of interest
  
  groupby_col : string
    Column in df to group by

  logp : string
    Column in df containing -log10(p-value)
  
  dim : tuple (default (20, 10))
    Dimensions of figure

  sig_pval_line : bool (default False)
    Whether to add line signifying significance cutoff
  
  sig_pval : float (default 0.05)
    Significant p-value cutoff
  
  axxlabel : string (default None)
    x-axis label

  axylabel : string (default None)
    y-axis label for plot

  axlabelfontsize : int or float (default 9)
    Font size of axes labels

  axlabelfontname : str (default "DejaVu Sans")
    Font of axes labels
  
  axtickfontsize : int or float (default 9)
    Font size of axes ticks (e.g. font size for each category)

  axticknfontname : string (default 'DejaVu Sans')
    Font of axes ticks (e.g. font for each category)

  ar : int or float (default 90)
    Rotation of x-axis ticks. Default perpendicular

  dotsize : int or float (default 8)
    Size of dots

  figtitle : string (default 'Manhattan Plot')
    Title of plot
  
  annotate : bool (default True)
    Whether to annotate dots (e.g. phenotypes) of interest
    Requires an 'annotate' column in df

  annotatefontsize : int or float (default 20)
    Font size of annotated dots of interest

  expand_text : array-like (default (1.05, 1.2))
    adjust_text parameter
    Default value is default value for adjust_text method
    From adjust_text docstring:
    a tuple/list/... with 2 multipliers (x, y) by which to expand the
    bounding box of texts when repelling them from each other.

  expand_points : array-like (default (1.05, 1.2))
    adjust_text parameter
    Default value is default value for adjust_text method
    From adjust_text docstring:
    a tuple/list/... with 2 multipliers (x, y) by which to expand the
    bounding box of texts when repelling them from points.

  autoalign : str or bool (default True)
    adjust_text parameter
    From adjust_text docstring:
    Direction in wich the best alignement will be determined
    - 'xy' or True, best alignment of all texts determined in all
      directions automatically before running the iterative adjustment
      (overriding va and ha),
    - 'x', will only align horizontally,
    - 'y', will only align vertically,
    - False, do nothing (i.e. preserve va and ha)
    
    Refer to adjust_text documentation for va and ha context
  
  save : bool (default False)
    Whether to save figure

  filename : str (default 'Manhattan_Plot.pdf')
    File name if saving figure
    Must include file path in filename
  
  """
  
  # _x denotes default x-axis label
  _x, _y = 'Phecode Category', r'$ -log_{10}(P)$'

  # Don't annotate y-axis for Fig 1, where there will be no annotation of text
  if annotate == False:
      _y = None

  # Sort DataFrame by groupby_col in alphabetical order
  df = df.sort_values(groupby_col)

  df['ind'] = range(len(df))
  df_group = df.groupby(groupby_col)

  
  # Colors that will be used for each category
  rand_colors = ('#a7414a', 
                 '#282726', 
                 '#6a8a82', 
                 '#a37c27', 
                 '#563838', 
                 '#0584f2', 
                 '#f28a30', 
                 '#f05837',
                 '#6465a5', 
                 '#00743f', 
                 '#be9063', 
                 '#de8cf0', 
                 '#888c46', 
                 '#c0334d', 
                 '#270101', 
                 '#8d2f23',
                 '#ee6c81', 
                 '#65734b', 
                 '#14325c', 
                 '#704307', 
                 '#b5b3be', 
                 '#f67280', 
                 '#ffd082', 
                 '#ffd800',
                 '#ad62aa', 
                 '#21bf73', 
                 '#a0855b', 
                 '#5edfff', 
                 '#08ffc8', 
                 '#ca3e47', 
                 '#c9753d', 
                 '#6c5ce7')
  
  color_list = rand_colors[:df[groupby_col].nunique()]

  xlabels = []
  xticks = []

  # Create matplotlib figure
  fig, ax = plt.subplots(figsize=dim)
  fig.tight_layout()
      
  # %%%% CREATE MANHATTAN PLOT %%%%
  
  # Variable i specifies index of color used (from rand_colors variable) for each category
  i = 0

  # List of which phenotypes to annotate
  texts = []

  # Kwargs for annotation
  kwargs_annotate = dict(color='k', fontweight='medium')
  
  # Populate scatter plot with -log10 p-value for each category
  for category, category_df in df.groupby(groupby_col):
      ax.scatter(category_df['ind'], category_df[logp], color=color_list[i], s=dotsize)

      # Add annotation
      # source: https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas-dataframe/39374693
      if annotate:
          for idx, row in category_df.iterrows():
              if row['annotate'] == 1:
                  texts.append(ax.annotate(row['phenotype'], 
                                   (row['ind'], row[logp]), 
                                   fontsize=annotatefontsize, 
                                   **kwargs_annotate))
      
      # Place xlabel and associated xtick in the middle of each category 
      category_df_max_ind = category_df['ind'].iloc[-1]
      category_df_min_ind = category_df['ind'].iloc[0]
      
      xlabels.append(category)
      xticks.append((category_df_max_ind - (category_df_max_ind - category_df_min_ind) / 2))

      i += 1

  # Apply adjust_text 
  if len(texts) > 0: 
    adjust_text(texts,
                arrowprops=dict(arrowstyle="-", 
                                color='k'),
                autoalign=autoalign,
                expand_text=expand_text,
                expand_points=expand_points,
                ax=ax)

  # add significant p-value cutoff line
  if sig_pval_line is True:
      ax.axhline(y=-np.log10(sig_pval), linestyle='--', color='#7d7d7d', linewidth=1)

  # %%%% FORMAT %%%%
  # No margins for x and y - dots can go up to edges of plot
  ax.margins(x=0, y=0)

  # Add xticks and associated labels to bottom subplot
  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabels, fontsize=axtickfontsize, rotation=ar)
  ax.tick_params(axis='y', labelsize=axlabelfontsize-8)

  # yticks label 
  ax.tick_params(axis='y', labelsize=axlabelfontsize-8)

  # y-axis max limit
  # ref: https://stackoverflow.com/questions/26454649/python-round-up-to-the-nearest-ten
  ylim_max = math.ceil(df[logp].max() / 10.0) * 10
  ax.set_ylim([0, ylim_max])

  if axxlabel:
      _x = axxlabel
  if axylabel:
      _y = axylabel
  # x-axis label
  ax.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
  # y-axis label
  ax.set_ylabel(_y, fontsize=axlabelfontsize, fontname=axlabelfontname)
  
  # Set title
  kwargs_title = dict(transform=ax.transAxes, linespacing=1, fontweight=650)
  ax.set_title(figtitle, fontsize=axlabelfontsize+2, **kwargs_title)

  # %%%% SAVE %%%%
  if save:
    fig.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')

  return fig

# COMMAND ----------

def mhat_RE(icd10_mapping, # 20230104 Note: Would likely want icd10_mapping=codemap3 here
            df="dataframe", 
            chromo=None, 
            logp=None, 
            color=None, 
            dim=(10,10), 
            rows=None,
            columns=None,
            nrowstop=None, # number of rows for top subplot
            nrowsmid=None, # number of rows for middle subplot
            topmin=None, # min y-axis value for top subplot
            topmax=None, # max y-axis value for top subplot
            mainmin=None, # min y-axis value for main subplot
            mainmax=None, # max y-axis value for main subplot
            r=300, 
            ar=90, 
            gwas_sign_line=False,
            gwasp=5E-08, 
            dotsize=8, 
            markeridcol=None, 
            markernames=None, 
            gfont=8, 
            valpha=1, 
            show=False, 
            figtype='pdf',
            axxlabel=None, 
            axylabel=None, 
            axlabelfontsize=9, 
            axlabelfontname="DejaVu Sans", 
            axtickfontsize=9, 
            figtitle='manhattan plot',
            textcolor='k', # text annotation color 
            axtickfontname="DejaVu Sans", 
            ylm=None, 
            gstyle=1, 
            yskip=1, 
            plotlabelrotation=0, 
            figname='miami', 
            invert=False,       
            fig=None, 
            ax=None,
            annotate=True,
            annotatefontsize=20,
            expand_text=(1.05, 1.2), # expand_text, expand_points, and autoalign set to adjust_text's default values
            expand_points=(1.05, 1.2),
            autoalign=True,
            invisible_ticks=False,
            suffix='',
            axescolors='k',
            overlap=None): # Phenotypes in overlap set will be bolded to indicate common top phenotypes among all groups
        
  # 20210823 Added icd10_mapping parameter to add in the icd-10 names

  # _y denotes what will be in y-axis
  _x, _y = 'Chromosomes', r'$ -log_{10}(P)$'

  # Don't annotate y-axis for Fig 1, where there will be no annotation of text
  if annotate == False:
      _y = None

  # tpval1 corresponds to -log10_pvalue of first group
  df['tpval'] = df[logp]
  df = df.sort_values(chromo)

  df['ind'] = range(len(df))
  df_group = df.groupby(chromo)

  rand_colors = ('#a7414a', 
                  '#282726', 
                  '#6a8a82', 
                  '#a37c27', 
                  '#563838', 
                  '#0584f2', 
                  '#f28a30', 
                  '#f05837',
                  '#6465a5', 
                  '#00743f', 
                  '#be9063', 
                  '#de8cf0', 
                  '#888c46', 
                  '#c0334d', 
                  '#270101', 
                  '#8d2f23',
                  '#ee6c81', 
                  '#65734b', 
                  '#14325c', 
                  '#704307', 
                  '#b5b3be', 
                  '#f67280', 
                  '#ffd082', 
                  '#ffd800',
                  '#ad62aa', 
                  '#21bf73', 
                  '#a0855b', 
                  '#5edfff', 
                  '#08ffc8', 
                  '#ca3e47', 
                  '#c9753d', 
                  '#6c5ce7')

  print(df[chromo].unique())
  color_list = rand_colors[:df[chromo].nunique()]

  xlabels = []
  xticks = []

  if fig is None:
      fig = plt.figure(figsize=dim)
      fig.tight_layout()
      
  rows = rows
  columns = columns

  # grid0 for y axis, grid1 for miami plot
  grid0 = plt.GridSpec(rows, columns, left=0.50, right=0.55) #  wspace = .25, hspace = .25, 
  grid1 = plt.GridSpec(rows, columns, hspace=0)

  ax0 = plt.subplot(grid0[:, 0])
  ax0.axis('off')

  i = 0
  texts = []
  kwargs1 = dict(color=textcolor, fontweight='medium')
  kwargs2 = dict(color='k', fontweight='heavy')
  for label, df1 in df.groupby(chromo):
      ax = plt.subplot(grid1[(nrowstop+1):, 1])
      ax.scatter(df1['ind'], df1['tpval'], color=color_list[i], s=dotsize)
      
      
      # Add annotation (20220111)
      # source: https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas-dataframe/39374693
      if annotate:
          for idx, row in df1.iterrows():
              if row['annotate'+suffix] == 1:
                  if idx in overlap:
                      texts.append(ax.annotate(idx, (row['ind'], row['tpval']), fontsize=annotatefontsize, **kwargs2))
                  else:
                      texts.append(ax.annotate(idx, (row['ind'], row['tpval']), fontsize=annotatefontsize, **kwargs1))
                  
              
      d = .007  # how big to make the diagonal lines in axes coordinates
      # arguments to pass to plot, just so we don't keep repeating them
      kwargs = dict(transform=ax.transAxes, color=axescolors, clip_on=False, linewidth=1)
      ax.plot((-d, +d), (1, 1), **kwargs)  # bottom-left diagonal
      ax.plot((1 - d, 1 + d), (1, 1), **kwargs)  # bottom-right diagonal
      
      # Add line breaks (20220112)
      # Plot the same data above on both additional axes
      
      # ax2 is top subplot
      ax2 = plt.subplot(grid1[0:nrowstop, 1])
      ax2.scatter(df1['ind'], df1['tpval'], color=color_list[i], s=dotsize)
      
      
      kwargs.update(transform=ax2.transAxes)
      ax2.plot((-d, +d), (0, 0), **kwargs)        # top-left diagonal
      ax2.plot((1 - d, 1 + d), (0, 0), **kwargs)  # top-right diagonal
      
      df1_max_ind = df1['ind'].iloc[-1]
      df1_min_ind = df1['ind'].iloc[0]
      xlabels.append(label)
      xticks.append((df1_max_ind - (df1_max_ind - df1_min_ind) / 2))
      
      i += 1

  adjust_text(texts,
              arrowprops=dict(arrowstyle="-", 
                              color='k'),
              autoalign=autoalign,
              expand_text=expand_text,
              expand_points=expand_points,
              ax=ax)
      
  ax.axhline(y=0, color='#7d7d7d', linewidth=.5, zorder=0)

  # 20210823 Change xlabels to ICD-10 names
  xlabels = icd10_mapping

  # add GWAS significant line
  if gwas_sign_line is True:
      ax.axhline(y=np.log10(gwasp), linestyle='--', color='#7d7d7d', linewidth=1)
      ax.axhline(y=-np.log10(gwasp), linestyle='--', color='#7d7d7d', linewidth=1)
  if markernames is not None:
      marker.geneplot_mhat_logp(df, 
                                markeridcol, 
                                chromo, 
                                'tpval', 
                                gwasp, 
                                markernames, 
                                gfont, 
                                gstyle, 
                                ax=ax, 
                                plotlabelrotation=plotlabelrotation)

  ax.margins(x=0)
  ax.margins(y=0)
  ax.set_xticks(xticks)
  ax.set_yticks(np.arange(mainmin, (mainmax+10), 10)) # set yticks for bottom subplot here
  ax.tick_params(axis='y', labelsize=axlabelfontsize-8)
  ax.set_ylim([mainmin, mainmax]) # limit for bottom subplot here
  ax.set_xticklabels(xlabels, fontsize=axtickfontsize, rotation=ar)
  ax.spines['top'].set_visible(False)

  ax2.margins(x=0)
  ax2.margins(y=0)
  ax2.set_yticks(np.arange(topmin, (topmax+10), 10)) # set yticks for top subplot here
  ax2.tick_params(axis='y', labelsize=axlabelfontsize-8)
  ax2.set_ylim([topmin, topmax]) # limit for top subplot here
  ax2.spines['bottom'].set_visible(False)
  ax2.xaxis.set_visible(False)
  kwargs=dict(transform=ax2.transAxes, linespacing=1, fontweight=650)
  ax2.set_title(figtitle, fontsize=axlabelfontsize+2, **kwargs)

  (ymin, ymax) = (0, topmax+10)
  ax0.set_ylim([ymin, ymax])

  ax0.text(0.5,ymax/2,_y, fontsize=axlabelfontsize, fontname=axlabelfontname, rotation = 90, va='center')

  if axxlabel:
      _x = axxlabel
  if axylabel:
      _y = axylabel
  ax.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
  ax.get_yaxis().get_label().set_visible(False)

  # 20220301 Invisible ticks for Fig 1; also changing axes colors
  # https://www.delftstack.com/howto/matplotlib/how-to-hide-axis-text-ticks-and-or-tick-labels-in-matplotlib/
  if invisible_ticks:
      ax.yaxis.set_visible(False)
      ax.xaxis.set_visible(False)
      ax2.yaxis.set_visible(False)
      ax2.xaxis.set_visible(False)
      
      # 20220411 Change spine color to specific identified race and ethnicity:
      # https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis
      ax.spines['bottom'].set_color(axescolors)
      ax.spines['top'].set_color(axescolors) 
      ax.spines['right'].set_color(axescolors)
      ax.spines['left'].set_color(axescolors)

      ax2.spines['bottom'].set_color(axescolors)
      ax2.spines['top'].set_color(axescolors) 
      ax2.spines['right'].set_color(axescolors)
      ax2.spines['left'].set_color(axescolors)
      
      # 20220411 Set line width to be thicker:
      # https://stackoverflow.com/questions/2553521/setting-axes-linewidth-without-changing-the-rcparams-global-dict
      # change all spines
      for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(8)
          ax2.spines[axis].set_linewidth(8)
          
      # 20220411 Set line width to be thicker for the cutoffs
      kwargs = dict(transform=ax.transAxes, color=axescolors, clip_on=False, linewidth=8)
      ax.plot((-d, +d), (1, 1), **kwargs)  # bottom-left diagonal
      ax.plot((1 - d, 1 + d), (1, 1), **kwargs)  # bottom-right diagonal
      
      # Add line breaks (20220112)
      # Plot the same data above on both additional axes
      
      # ax2 is top subplot
      kwargs.update(transform=ax2.transAxes)
      ax2.plot((-d, +d), (0, 0), **kwargs)        # top-left diagonal
      ax2.plot((1 - d, 1 + d), (0, 0), **kwargs)  # top-right diagonal
      
  general.get_figure(show, r, figtype, figname)
  return fig, ax

# COMMAND ----------

def mhat_two_line_breaks(df, # pandas DataFrame
                         groupby_col=None, # column to group by
                         logp=None, # df column containing -log 10 of the p-value
                         
                         # Figure parameters
                         dim=(40,10), 

                         rows=None, # total number of rows; should equal nrowstop + nrowsmid + nrowsbot (not defined  function because not required for it to run) + 2
                         columns=2, # total number of columns - one for y-axis, the other for manhattan plot
                         nrowstop=None, # number of rows for top subplot
                         nrowsmid=None, # number of rows for middle subplot
                         topmin=None, # min y-axis value for top subplot
                         topmax=None, # max y-axis value for top subplot
                         midmin=None, # min y-axis value for middle subplot
                         midmax=None, # max y-axis value for middle subplot
                         botmin=None, # min y-axis value for bottom subplot
                         botmax=None, # max y-axis value for bottom subplot
                         
                         sig_pval_line=False, # bool that adds significance cutoff line
                         sig_pval=0.05, # significant p-value cutoff
                         
                         axxlabel=None, # x-axis label for bottom subplot 
                         axylabel=None, # y-axis label (used?)
                         axlabelfontsize=9, # x-axis label fontsize for bottom subplot
                         axlabelfontname="DejaVu Sans", # font used
                         axtickfontsize=9, # x-axis tick label fontsize
                         axtickfontname="DejaVu Sans", # font for xticks 
                         ar=90, # How much to rotate phecode category name on x axis
                         
                         dotsize=8, # size of markers
                         figtitle='Manhattan Plot', # figure title
                         
                         # Annotation parameters
                         annotate=True, # Whether to annotate
                         annotatefontsize=20, # annotation font size
                         expand_text=(1.05, 1.2), # expand_text, expand_points, and autoalign set to adjust_text's default values
                         expand_points=(1.05, 1.2),
                         autoalign=True, 
                         
                         # Save parameters
                         save=False,
                         filename='Manhattan_Plot.pdf'): 

  """
  Makes Manhattan plot of data

  Parameters
  __________
  df : pandas DataFrame
    Contains data of interest
  
  groupby_col : string
    Column in df to group by

  logp : string
    Column in df containing -log10(p-value)
  
  dim : tuple (default (40, 10))
    Dimensions of figure

  rows : int
    Number of rows in figure
    Equal to number of rows including line breaks
    For example, if bottom subplot has 4 rows, middle subplot has 2 rows, and top subplot has 1 row
    The total number of rows with two line breaks = 4 (# rows bottom subplot) + 
                                                    2 (# rows middle subplot) + 
                                                    1 (# rows top subplot) +
                                                    2 (# line breaks)
                                                    = 9

  columns : int (default 2)
    Number of columns in figure
    Set to 2. One column for y-axis, the other column for Manhattan 

  nrowstop : int
    Number of rows comprising top subplot
  
  nrowsmid : int
    Number of rows comprising middle subplot

  topmin : int
    y-axis minimum value for top subplot
  
  topmax : int
    y-axis maximum value for top subplot

  midmin : int
    y-axis minimum value for middle subplot
  
  midmax : int
    y-axis maximum value for middle subplot

  botmin : int
    y-axis minimum value for bottom subplot
  
  botmax : int
    y-axis maximum value for bottom subplot

  sig_pval_line : bool (default False)
    Whether to add line signifying significance cutoff
  
  sig_pval : float (default 0.05)
    Significant p-value cutoff
  
  axxlabel : string (default None)
    x-axis label, specifically used for bottom subplot

  axylabel : string (default None)
    y-axis label for plot, generally not needed

  axlabelfontsize : int or float (default 9)
    Font size of axes labels

  axlabelfontname : str (default "DejaVu Sans")
    Font of axes labels
  
  axtickfontsize : int or float (default 9)
    Font size of axes ticks (e.g. font size for each category)

  axticknfontname : string (default 'DejaVu Sans')
    Font of axes ticks (e.g. font for each category)

  ar : int or float (default 90)
    Rotation of x-axis ticks. Default perpendicular

  dotsize : int or float (default 8)
    Size of dots

  figtitle : string (default 'Manhattan Plot')
    Title of plot, specifically used for top subplot
  
  annotate : bool (default False)
    Whether to annotate dots (e.g. phenotypes) of interest
    Requires an 'annotate' column in df

  annotatefontsize : int or float (default 20)
    Font size of annotated dots of interest

  expand_text : array-like (default (1.05, 1.2))
    adjust_text parameter
    Default value is default value for adjust_text method
    From adjust_text docstring:
    a tuple/list/... with 2 multipliers (x, y) by which to expand the
    bounding box of texts when repelling them from each other.

  expand_points : array-like (default (1.05, 1.2))
    adjust_text parameter
    Default value is default value for adjust_text method
    From adjust_text docstring:
    a tuple/list/... with 2 multipliers (x, y) by which to expand the
    bounding box of texts when repelling them from points.

  autoalign : str or bool (default True)
    adjust_text parameter
    From adjust_text docstring:
    Direction in wich the best alignement will be determined
    - 'xy' or True, best alignment of all texts determined in all
      directions automatically before running the iterative adjustment
      (overriding va and ha),
    - 'x', will only align horizontally,
    - 'y', will only align vertically,
    - False, do nothing (i.e. preserve va and ha)
    
    Refer to adjust_text documentation for va and ha context
  
  save : bool (default False)
    Whether to save figure

  filename : str (default 'Manhattan_Plot.pdf')
    File name if saving figure
    Must include file path in filename
  
  """
  
  # _x denotes default x-axis label
  _x, _y = 'Phecode Category', r'$ -log_{10}(P)$'

  # Don't annotate y-axis for Fig 1, where there will be no annotation of text
  if annotate == False:
      _y = None

  # Sort DataFrame by groupby_col in alphabetical order
  df = df.sort_values(groupby_col)

  df['ind'] = range(len(df))
  df_group = df.groupby(groupby_col)

  
  # Colors that will be used for each category
  rand_colors = ('#a7414a', 
                 '#282726', 
                 '#6a8a82', 
                 '#a37c27', 
                 '#563838', 
                 '#0584f2', 
                 '#f28a30', 
                 '#f05837',
                 '#6465a5', 
                 '#00743f', 
                 '#be9063', 
                 '#de8cf0', 
                 '#888c46', 
                 '#c0334d', 
                 '#270101', 
                 '#8d2f23',
                 '#ee6c81', 
                 '#65734b', 
                 '#14325c', 
                 '#704307', 
                 '#b5b3be', 
                 '#f67280', 
                 '#ffd082', 
                 '#ffd800',
                 '#ad62aa', 
                 '#21bf73', 
                 '#a0855b', 
                 '#5edfff', 
                 '#08ffc8', 
                 '#ca3e47', 
                 '#c9753d', 
                 '#6c5ce7')
  
  color_list = rand_colors[:df[groupby_col].nunique()]

  xlabels = []
  xticks = []

  # Create matplotlib figure
  fig = plt.figure(figsize=dim)
  fig.tight_layout()
      
  rows = rows
  columns = columns

  # grid0 for y-axis plot, grid1 for main manhattan plot
  grid0 = plt.GridSpec(rows, columns, left=0.50, right=0.55)  
  grid1 = plt.GridSpec(rows, columns, hspace=0)

  # y-axis subplot variable
  ax_yaxis = plt.subplot(grid0[:, 0])
  ax_yaxis.axis('off')

  # %%%% CREATE MANHATTAN PLOT %%%%
  
  # Variable i specifies index of color used (from rand_colors variable) for each category
  i = 0

  # List of which phenotypes to annotate
  texts_top = []
  texts_mid = []
  texts_bottom = []
  
  # Kwargs for annotation
  kwargs_annotate = dict(color='k', fontweight='medium')
  
  # Populate scatter plot with -log10 p-value for each category
  # NOTE: slicing goes from TOP of the subplot to BOTTOM of the subplot
  
  for category, category_df in df.groupby(groupby_col):
      # *** Top subplot ***
      ax_top = plt.subplot(grid1[0:nrowstop, 1])
      ax_top.scatter(category_df['ind'], category_df[logp], color=color_list[i], s=dotsize)

      # Add annotation
      # source: https://stackoverflow.com/questions/15910019/annotate-data-points-while-plotting-from-pandas-dataframe/39374693
      if annotate:
          for idx, row in category_df.iterrows():
              if row['annotate'] == 1:
                  texts_top.append(ax_top.annotate(row['phenotype'], 
                                   (row['ind'], row[logp]), 
                                   fontsize=annotatefontsize, 
                                   **kwargs_annotate))
      
      # *** Add line breaks ***

      # size of line break lines 
      d = 0.007  

      # Kwargs to pass to each subplot
      kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)

      # Add line breaks for top subplot
      kwargs.update(transform=ax_top.transAxes)
      ax_top.plot((-d, +d), (0, 0), **kwargs)        # bottom-left line break line
      ax_top.plot((1 - d, 1 + d), (0, 0), **kwargs)  # bottom-right line break line

      # *** Middle subplot ***
      ax_middle = plt.subplot(grid1[(nrowstop+1):(nrowstop+nrowsmid+1), 1]) 
      ax_middle.scatter(category_df['ind'], category_df[logp], color=color_list[i], s=dotsize)

      # Add annotation
      if annotate:
          for idx, row in category_df.iterrows():
              if row['annotate'] == 1:
                  texts_mid.append(ax_middle.annotate(row['phenotype'], 
                                   (row['ind'], row[logp]), 
                                   fontsize=annotatefontsize, **kwargs_annotate))
      
      # Add line breaks for middle subplot
      kwargs.update(transform=ax_middle.transAxes)
      ax_middle.plot((-d, +d), (0, 0), **kwargs)        # bottom-left line break line #(-d, +d)
      ax_middle.plot((1 - d, 1 + d), (0, 0), **kwargs)  # bottom-right line break line
      ax_middle.plot((-d, +d), (1, 1), **kwargs)        # top-left line break line
      ax_middle.plot((1 - d, 1 + d), (1, 1), **kwargs)  # top-right line break line
      
      # *** Bottom subplot ***
      ax_bottom = plt.subplot(grid1[(nrowstop+nrowsmid+2):, 1]) # (nrowstop+nrowsmid+2)
      ax_bottom.scatter(category_df['ind'], category_df[logp], color=color_list[i], s=dotsize)
      
      # Add annotation 
      if annotate:
          for idx, row in category_df.iterrows():
              if row['annotate'] == 1:
                  texts_bottom.append(ax_bottom.annotate(row['phenotype'], 
                                          (row['ind'], row[logp]), 
                                          fontsize=annotatefontsize, **kwargs_annotate))
                  
      # Add line breaks for bottom subplot
      kwargs.update(transform=ax_bottom.transAxes)
      ax_bottom.plot((-d, +d), (1, 1), **kwargs)        # top-left line break
      ax_bottom.plot((1 - d, 1 + d), (1, 1), **kwargs)  # top-right line break
      
      # Place xlabel and associated xtick in the middle of each category in manhattan subplot
      category_df_max_ind = category_df['ind'].iloc[-1]
      category_df_min_ind = category_df['ind'].iloc[0]
      
      xlabels.append(category)
      xticks.append((category_df_max_ind - (category_df_max_ind - category_df_min_ind) / 2))

      i += 1

  # Apply adjust_text to bottom subplot
  if len(texts_bottom) > 0: 
    adjust_text(texts_bottom,
                arrowprops=dict(arrowstyle="-", 
                                color='k'),
                autoalign=autoalign,
                expand_text=expand_text,
                expand_points=expand_points,
                ax=ax_bottom)
  
  ax_bottom.axhline(y=0, color='#7d7d7d', linewidth=.5, zorder=0)

  # add significant p-value cutoff line
  if sig_pval_line is True:
      ax_bottom.axhline(y=-np.log10(sig_pval), linestyle='--', color='#7d7d7d', linewidth=1)

  # %%%% FORMAT MAIN MANHATTAN SUBPLOT %%%%
  # *** Format top subplot ***
  # No margins for x and y - dots can go up to edges of plot
  ax_top.margins(x=0, y=0)

  # y-axis min and max limit
  ax_top.set_ylim([topmin, topmax]) 

  # Set yticks range 
  ax_top.set_yticks(np.arange(topmin, (topmax+10), 10)) 

  # yticks label 
  ax_top.tick_params(axis='y', labelsize=axlabelfontsize-8)
  
  # Bottom line set to invisible
  ax_top.xaxis.set_visible(False)
  ax_top.spines['bottom'].set_visible(False)
  
  # Set title
  kwargs_title = dict(transform=ax_top.transAxes, linespacing=1, fontweight=650)
  ax_top.set_title(figtitle, fontsize=axlabelfontsize+2, **kwargs_title)

  # *** Format middle subplot ***
  ax_middle.margins(x=0, y=0)

  ax_middle.set_ylim([midmin, midmax]) 

  ax_middle.set_yticks(np.arange(midmin, (midmax+10), 10)) 
  
  ax_middle.tick_params(axis='y', labelsize=axlabelfontsize-8)
  
  ax_middle.xaxis.set_visible(False)
  ax_middle.spines['top'].set_visible(False)
  ax_middle.spines['bottom'].set_visible(False)
  
  # *** Format bottom subplot ***
  ax_bottom.margins(x=0, y=0)
  
  # Add xticks and associated labels to bottom subplot
  ax_bottom.set_xticks(xticks)
  ax_bottom.set_xticklabels(xlabels, fontsize=axtickfontsize, rotation=ar)
  ax_bottom.tick_params(axis='y', labelsize=axlabelfontsize-8)

  ax_bottom.set_ylim([botmin, botmax]) 

  ax_bottom.set_yticks(np.arange(botmin, (botmax+10), 10)) 

  ax_bottom.spines['top'].set_visible(False)

  
  if axxlabel:
      _x = axxlabel
  if axylabel:
      _y = axylabel
  # x-axis label
  ax_bottom.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
  
  # Set y-axis invisible for manhattan subplot
  ax_bottom.get_yaxis().get_label().set_visible(False)

  # %%%% FORMAT Y-AXIS OF MANHATTAN PLOT
  (ymin, ymax) = (0, topmax+10)
  ax_yaxis.set_ylim([ymin, ymax])
  ax_yaxis.text(0.5, ymax/2, _y, fontsize=axlabelfontsize, fontname=axlabelfontname, rotation = 90, va='center')

  # %%%% SAVE %%%%
  if save:
    fig.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')

  return fig

# COMMAND ----------

# For log-log plots
def set_axes_bounds(df_comp, model_reference, model_compared):
  """
  Sets axes bounds to be equal on x and y axis

  Parameters
  __________
  df_comp : pandas DataFrame
    Contains comparison of reference model to model of interest

  model_reference : string
    Reference model. Likely primary model  

  model_compared : string
    Which model is being compared to reference model

  Returns
  _______
  axes_min : float
    Minimum axes value for both x and y axes

  axes_max : float
    Maxmimum axes value for both x and y axes
  """

  # Set axes bounds
  # x-axis (primary)
  xmin_pre = df_comp['log10_OR_has_phenotype_' + model_reference].min()
  xmin = np.floor(xmin_pre)

  xmax_pre = df_comp['log10_OR_has_phenotype_' + model_reference].max()
  xmax = np.ceil(xmax_pre)

  # y-axis (crude)
  ymin_pre = df_comp['log10_OR_has_phenotype_' + model_compared].min()
  ymin = np.floor(ymin_pre)

  ymax_pre = df_comp['log10_OR_has_phenotype_' + model_compared].max()
  ymax = np.ceil(ymax_pre)

  # Get axes limits for both x and y
  if xmin < ymin:
    axes_min = xmin
  else:
    axes_min = ymin

  if xmax > ymax:
    axes_max = xmax
  else:
    axes_max = ymax

  # Set min and max values to be equal
  if np.abs(axes_min) > axes_max:
    axes_max = np.abs(axes_min)
  else:
    axes_min = axes_max * -1

  return axes_min, axes_max

# COMMAND ----------

# For ln-ln plots
def set_axes_bounds_ln(df_comp, model_reference, model_compared):
  """
  Sets axes bounds to be equal on x and y axis

  Parameters
  __________
  df_comp : pandas DataFrame
    Contains comparison of reference model to model of interest

  model_reference : string
    Reference model. Likely primary model  

  model_compared : string
    Which model is being compared to reference model

  Returns
  _______
  axes_min : float
    Minimum axes value for both x and y axes

  axes_max : float
    Maxmimum axes value for both x and y axes
  """

  # Set axes bounds
  # x-axis (primary)
  xmin_pre = df_comp['ln_OR_has_phenotype_' + model_reference].min()
  xmin = np.floor(xmin_pre)

  xmax_pre = df_comp['ln_OR_has_phenotype_' + model_reference].max()
  xmax = np.ceil(xmax_pre)

  # y-axis (crude)
  ymin_pre = df_comp['ln_OR_has_phenotype_' + model_compared].min()
  ymin = np.floor(ymin_pre)

  ymax_pre = df_comp['ln_OR_has_phenotype_' + model_compared].max()
  ymax = np.ceil(ymax_pre)

  # Get axes limits for both x and y
  if xmin < ymin:
    axes_min = xmin
  else:
    axes_min = ymin

  if xmax > ymax:
    axes_max = xmax
  else:
    axes_max = ymax

  # Set min and max values to be equal
  if np.abs(axes_min) > axes_max:
    axes_max = np.abs(axes_min)
  else:
    axes_min = axes_max * -1

  return axes_min, axes_max

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing 04_UMAP Notebooks...**

# COMMAND ----------

# MAGIC %md
# MAGIC *Importing non plotting functions for 04_UMAP Notebooks...*

# COMMAND ----------

def prepare_diagnosis_data(diag_all_spark, file_name_diag, demo_spark):
  """
  This removes phecode-corresponding phenotypes not associated with a phecode category
  
  Parameters
  __________
  diag_all_spark : spark DataFrame
    Contains ICD-, SNOMED- and phecode- based diagnoses for patients 
    
  Returns
  _______
  diag : pandas DataFrame
    Contains corresponding phecodes, phenotypes, and associated phecode_category of
    the diagnoses for patients 
  """

  print(color.BOLD + '%%% Preparing data for ' + file_name_diag + ' %%%' + color.END)
  print()
  diag_all = diag_all_spark.toPandas()
  print(f"Shape of {file_name_diag} before removing null icd10-inspired chapters and diagnoses that are not conditions is {diag_all.shape}.")

  # Only keep diagnoses that are conditions
  diag = diag_all[diag_all['ICD_domain_id'] == 'Condition']

  # Only keep diagnoses mapped to phecodes that are organized into ICD-10 inspired chapters
  diag = diag_all[~diag_all['phecode_category'].isnull()]
  print(f"Shape of {file_name_diag} after removing null icd10-inspired chapters and diagnoses that are not conditions is {diag.shape}.")

  # Merge demo info to retain the remaining patients if needed (most pertinent for controls):
  
  # Convert demo_spark to pandas DataFrame
  print('\n')
  print("Merging patients who don't have any conditions back into the diagnoses dataframe...")
  demo = demo_spark.toPandas()
  
  diag = demo['person_id'].to_frame().merge(diag,
                                            how='left',
                                            on='person_id')
  
  print(f"Diagnosis shape after merging back demo: {diag.shape}")
  # ***** Clean data *****
  print('\n')
  print("Only keeping the following columns: person_id, phecode, phenotype, phecode_category")
  diag = diag[['person_id',
               'phecode',
               'phenotype',
               'phecode_category']].copy().drop_duplicates()
  
  print('\n')
  print(f"Number of patients is {diag['person_id'].nunique()}")
  print('\n')
  
  print(f"Final diagnosis shape: {diag.shape}")
  print('\n')
  
  print(color.BOLD + 'Done' + color.END)
  print('\n')
  
  return diag

# COMMAND ----------

def make_pivot_tables(diag, file_name_diag, n='phenotype'):
  """
  Makes matrix containing one-hot encoding of each phenotype that each patient has (or doesn't have)
  
  Parameters
  __________
  diag : pandas DataFrame
    Contains phecodes, phenotypes, and corresponding phecode_categorys for each patient

  file_name_diag : pandas DataFrame
    Name of diag DataFrame. Notes mi status and which model it is derived from
    
  n : string (default 'phenotype')
    What kind of diagnoses is being one hot encoded
  
  Returns
  _______
  pivot_table : pandas DataFrame
    Matrix containing one-hot encoding of each phenotype that each patient has (or doesn't have)
    Each column is a phenotype (1 if patient has phenotype, else 0)
    Each row is a patient
  """
  
  print(color.BOLD + '%%% Making pivot table for ' + file_name_diag + ' %%%' + color.END)
  
  pivot_table = pd.pivot_table(diag[[n, 'person_id']].drop_duplicates(), 
                              values=[n], index='person_id', columns=[n],
                              aggfunc=lambda x: 1 if len(x)>0 else 0, 
                              fill_value=0)
  
  if 'mi' in file_name_diag:
    pivot_table['male infertility status'] = 1
  else:
    pivot_table['male infertility status'] = 0
  
  print(f"Shape of pivot table: {pivot_table.shape}")
  display(pivot_table.head(3))
  
  print('\n')
  print(color.BOLD + 'Done' + color.END)
  print('\n')
  
  return pivot_table

# COMMAND ----------

def dimensionality_reduction(diag, 
                             file_name, 
                             metric='cosine', 
                             random_state=42,
                             low_memory=True,
                             verbose=True):
  """
  Performs dimensionality reduction of patient diagnoses (2D). Saves result as spark DataFrame.
  
  Parameters
  __________
  
  diag : pandas DataFrame
    Contains patients' diagnoses information in the form of phecode-corresponding phenotypes

  file_name : string
    File name corresponding to PS matching strategy 

  metric : string (default 'cosine')
    From https://umap-learn.readthedocs.io/en/latest/parameters.html:
    controls how distance is computed in the ambient space of the input data. Parameter of UMAP

  random_state : int
    Seed for consistent results. Parameter of UMAP.

  low_memory : bool (default True)
    From https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html
    Whether to pursue lower memory NNdescent. Parameter of UMAP
    
  verbose : bool (default True)
    From https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html
    Whether to print status data during the computation.
    
  Returns
  _______
  X : numpy array 
    Containing each patient's coordinates in UMAP
  """
  
  print(color.BOLD + f"Peforming dimensionality reduction for {file_name}..." + color.END)
 
  X = diag.drop('male infertility status', axis=1).astype('int32')
  
  mapper = umap.UMAP(metric=metric, random_state=random_state, low_memory=low_memory, verbose=verbose).fit(X)
  
  X_embedded = mapper.transform(X)
  
  X_pd = pd.DataFrame(X_embedded)
  X_pd = X_pd.reset_index()
  
  X_spark = spark.createDataFrame(X_pd)
  
  display(X_spark.tail(5))
  print('Saving X as spark DataFrame...')
  X_spark.write.mode("overwrite").parquet("/mnt/ucsf-sarah-woldemariam-workspace/male_infertility/mi_logit_python/UMAP/" + file_name)
  print('Saved')
  
  return X_embedded

# COMMAND ----------

def make_X_embedded(X):
  """
  Makes X_embedded numpy array of patients' UMAP coordinates
  
  Parameters
  __________
  X : pandas DataFrame
    Contains patients' UMAP coordinates
    
  Returns
  _______
  X_embedded : numpy array
    Contains patients' UMAP coordinates
  """
  
  X = X.drop('index', axis=1).copy()
  
  # Turn into numpy array
  X_embedded = X.to_numpy()
  
  return X_embedded

# COMMAND ----------

# MAGIC %md
# MAGIC *Importing plotting functions for 04_UMAP Notebooks...*

# COMMAND ----------

def visualize_UMAP_data(X_embedded, y, feature, hue_order, palette='Set1', bbox_to_anchor=(1.5, 1.0), save=False, file_name=None, alpha=0.5, label_axes=True, figure_size=(10,8)):
  """
  Visualize patients' phenotypic profiles via UMAP, with hue based on demographic or some other type of feature
  
  Parameters
  __________
  X_embedded : numpy array
    2D numpy array 
    each row [x] corresponds to a patient
    first column [x][0] corresponds to first UMAP component
    second column [x][1] corresponds to second UMAP component
    
  y : series
    Contains demographic or other feature of interest associated with each patient
    
  feature : string
    Feature the UMAP is colored by
  
  hue_order : list
    Ordering of how patients are categorized for a given demographic or other feature of interest

  palette : string (default 'Set1')
    Color palette used for UMAP visualization
    
  bbox_to_anchor : 2-element tuple of floats (default (1.1, 1.0))
    Where to place legend
    
  save : bool (default False)
    Saves figure if True

  file_name : string (default None)
    Name of file if saving
    
  alpha : float (default 0.5)
    How transparent to make dots

  label_axes : bool (default True)
    Whether to label x and y axes as UMAP Component 1 and UMAP Component 2, respectively

  figure_size : tuple (default (10, 8))
    Set size of figure

  Returns
  _______
  Nothing
  """
  
  print(color.BOLD + f"UMAP based on {feature}" + color.END)
  with sns.color_palette("Set1"):
      fig = plt.figure(figsize=figure_size)
      indices = np.arange(X_embedded.shape[0])
      sns.scatterplot(x=X_embedded[indices , 0], 
                      y=X_embedded[indices , 1], 
                      hue=y[indices], 
                      s=20, 
                      linewidth=.0, alpha=alpha,
                      hue_order=hue_order,
                      palette=palette)
      ax = plt.gca()
      ax.set(xticks=[], yticks=[], facecolor='white')
      # https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis
      ax.spines['bottom'].set_color('k')
      ax.spines['top'].set_color('k') 
      ax.spines['right'].set_color('k')
      ax.spines['left'].set_color('k')

      plt.title(f"Phenotypes as Features, Colored by {feature}\n", fontweight='bold', fontsize=18)
      plt.legend(bbox_to_anchor=bbox_to_anchor, fontsize=14) 
      
      if label_axes:
        plt.xlabel('UMAP Component 1', fontsize=16)
        plt.ylabel('UMAP Component 2', fontsize=16)

  if save:
          plt.savefig(f"/dbfs/FileStore/{file_name}.pdf", format='pdf', dpi=300, bbox_inches='tight')

  plt.show()
  
  print('\n')

# COMMAND ----------

# UMAP violin plots
def make_UMAP_violin_plots(X_embedded, y_values, order=None, palette=None, figure_size=(10, 3), save=False, filename_UMAP_1=None, filename_UMAP_2=None):
  """
  Makes violin plots for UMAP components 1 and 2

  Parameters
  __________
  X_embedded : numpy array
    Contains patients' UMAP coordinates

  y_values : numpy array
    Contains patients' feature of interest

  figure_size : tuple (default (10, 3))
    Size of violin plots

  order : list
    Order of categories in violin plot
  
  palette : list OR string
    Colors of categories in violin plot
      Can specify colors in a list
      Alternatively, can specify color palette as string
    For mi status:
      palette=['hotpink', 'dodgerblue']

  save : bool (default False)
    Whether to save violin plots
  
  filename_UMAP_1 : string 
    File name of violin plot for UMAP Component 1

  filename_UMAP_2 : string 
    File name of violin plot for UMAP Component 2

  Returns
  _______
  Nothing
  """

  # UMAP Component 1
  fig, ax = plt.subplots(1, 1, figsize=figure_size)

  ax = sns.violinplot(x=X_embedded[:,0], 
                      y=y_values, 
                      order=order,
                      bw=.1,
                      palette=palette)
                  
  ax.set_yticklabels([])
  plt.xlabel('UMAP Component 1', fontsize=18)
  plt.tick_params(axis='x', which='major', labelsize=16)
  plt.tick_params(axis='y', which='both', left=False, labelbottom=False)

  if save: 
      plt.savefig(f"/dbfs/FileStore/{filename_UMAP_1}.pdf", format='pdf', dpi=300, bbox_inches='tight')

  plt.show()

  # UMAP Component 2  
  fig, ax = plt.subplots(1, 1, figsize=figure_size)

  ax = sns.violinplot(x=X_embedded[:,1], 
                      y=y_values, 
                      order=order,
                      bw=.1,
                      palette=palette)

  ax.set_yticklabels([])
  plt.xlabel('UMAP Component 2', fontsize=18)
  plt.xticks(rotation=270)
  plt.tick_params(axis='x', which='major', labelsize=16)
  plt.tick_params(axis='y', which='both', left=False, labelbottom=False)

  if save: 
      plt.savefig(f"/dbfs/FileStore/{filename_UMAP_2}.pdf", format='pdf', dpi=300, bbox_inches='tight')
  plt.show()

# COMMAND ----------

# Converting UMAP to dataframes for Kruskal-Wallis test:
def convert_UMAP_array_to_df(numpy_array):
    converted = pd.DataFrame(numpy_array, columns=['axis_1', 'axis_2'])
    return converted

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions imported.

# COMMAND ----------



