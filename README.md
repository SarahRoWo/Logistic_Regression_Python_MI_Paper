# Exploring male infertility patients' comorbidities using structured EHR data

# Updated July 17, 2025

**This version minimizes reliance on SQL querying, with most data wrangling and analyses performed in Python**

## Overview

This set of notebooks enables the analysis of comorbidities associated with male infertility using structured EHR data. First, we identified nonoverlapping patients with male infertility and patients who have had a vasectomy-related procedure (the control group). Second, patients' demographic features and diagnoses were extracted from the EHR. Third, diagnosis associations with male infertility were analyzed using logistic regression models that explored diagnosis associations < 6 months after male infertility diagnosis / vasectomy-related procedure and > 6 months after diagnosis/procedure (more information on these models will be provided below). Fourth, we visualized and compared results across logistic regression models. Finally, we used UMAP to visualize patients using numerous features.

## Description of notebooks

**MI_Functions notebook contains all functions used in analyses**

### 01_Cohort

This set of notebooks identifies male infertility and vasectomy patients and provides a table 1 comparing demographics

#### 01_All_Patients

This notebook identifies male infertility (case) and vasectomy (control) patients and their associated demographics (age, race category, ethnicity category, ADI category) before adding hospital utilization measures and before filtering for patients who have had their first male infertility diagnosis within 6 months of the database refresh. **It is important to know when the database has been last refreshed in order to determine the last day patient information has been recorded in the EHR.**

This notebook filters for patients who:
- are male-identified 
- are aged 18+ 
- have been treated in one of the following five UC centers: 
  - UC Davis
  - UC Irvine
  - UC Los Angeles
  - UC San Francisco
  - UC San Diego

#### 02_Retrieve_Tables

These are subsets of OMOP tables, filtering for rows including data on patients in the study

##### concept

Includes concept information related to patients' ICD diagnoses. Also includes concept information corresponding to gender_concept_id, race_concept_id, and ethnicity_concept_id

##### condition_occurrence

Includes patients' condition occurrence information 

##### measurement

Includes vasectomy patients' measurement information that's specifically related to vasectomy

##### observation

Includes patients' observation information

##### observation_adi

Includes patients' area deprivation index information (from OMOP observation table)

##### procedure_occurrence

Includes patients' procedure information

##### visit_occurrence

Includes patients' vist information

#### 03_Filter_Patients

##### Filter_mi_patients

This notebook filters out:
1. Patients with no non male infertility phenotypes
2. No associated area deprivation index
3. Patients who have received their male infertility diagnosis within 6 months of the database refresh (e.g., if database was last refreshed on June 30, 2023, won't consider patients who received a diagnosis within 6 months of this date, like January 1, 2023)

Additionally, this notebook obtains:
1. Total number of visits
2. Total length in the EHR in months
3. Date when patient first received a male infertility diagnosis
4. Patients' estimated age when they first received a male infertility diagnosis
5. The 6 month cutoff date - this is 6 months after the date patient first received a male infertility diagnosis
6. Number of visits before, after, or at the same date as the 6 month cutoff date
7. Length in the EHR in months before or after the 6 month cutoff date

##### Filter_vas_patients

This notebook filters out:
1. Patients with no phenotypes
2. No associated area deprivation index
3. Patients who had their first vasectomy-related concept recorded within 6 months of the database refresh

Additionally, this notebook obtains:
1. Total number of visits
2. Total length in the EHR in months
3. Date when patient first had vasectomy-related procedure or measurement recorded
4. Patients' estimated age when they first had vasectomy-related procedure or measurement recorded
5. The 6 month cutoff date - this is 6 months after the date patient first had vasectomy-related procedure or measurement recorded
6. Number of visits before, after, or at the same date as the 6 month cutoff date
7. Length in the EHR in months before or after the 6 month cutoff date

#### 04_TableOne_R

Creates Table 1. Compares demographic characteristics between male infertility patients and vasectomy patients.

### 02_Diagnoses

This set of notebooks acquires patient diagnoses.

#### 01_Obtain_ICD_SNOMED_diagnoses_MI_Vas

Obtains SNOMED and associated ICD diagnoses for male infertility and vasectomy patients. Specifies whether diagnoses occurred less than 6 months after first diagnosis/procedure, greater than 6 months after first diagnosis/procedure, or on the same date as 6 months after first diagnosis/procedure

#### 02_Obtain_phecodes

Obtains phecodes and phecode-corresponding phenotypes for each patients' diagnoses. Adds first phenotype start date and whether it occurred less than 6 months after first diagnosis/procedure, greater than 6 months after first diagnosis/procedure, or on the same date as 6 months after first diagnosis/procedure

### 03_Logistic_Regression

This set of notebooks runs logistic regression models along with comparing and visualizing results.

#### 01a_LogisticRegression_Before

Performs logistic regression phecode-corresponding phenotypes that first occurred less than 6 months after diagnosis/procedure. 

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (less than 6 months after diagnosis/procedure) + months in the EHR (less than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 01b_LogisticRegression_After

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 02a_Visualize_Enrichment_Analyses_Before

Visualizes volcano and Manhattan plots for the primary analysis for associations with phenotypes that first occurred less than 6 months after first diagnosis/procedure.

#### 02b_Visualize_Enrichment_Analyses_After

Visualizes volcano and Manhattan plots for the primary analysis for associations with phenotypes that first occurred greater than 6 months after first diagnosis/procedure.

#### 3a_Logit_LogLog_BH_Upset_Before

This notebook compares logistic regression analyses performed in 01a_LogisticRegression_Before

First, creates log-log plots comparing which phenotypes are significant in the primary analysis vs:
1. crude analysis
2. social determinants of health sensitivity analysis
3. hospital utilization sensitivity analysis

Second, creates upset plots comparing number of phenotypes that are significant between combinations of logistic regression analyses. Additionally, specifies which phenotypes are significant between combinations of logistic regression analyses.

#### 3b_Logit_LogLog_BH_Upset_After

This notebook compares logistic regression analyses performed in 01b_LogisticRegression_After

First, creates log-log plots comparing which phenotypes are significant in the primary analysis vs:
1. crude analysis
2. social determinants of health sensitivity analysis
3. hospital utilization sensitivity analysis

Second, creates upset plots comparing number of phenotypes that are significant between combinations of logistic regression analyses. Additionally, specifies which phenotypes are significant between combinations of logistic regression analyses.

#### 3c_Logit_LogLog_BH_Upset_Before_vs_After

This notebook compares the primary logistic regression analyses performed in 01a_LogisticRegression_Before vs 01b_LogisticRegression_After

First, creates log-log plots comparing which phenotypes are significant in the primary analysis from 01a_LogisticRegression_Before vs the primary analysis from 01b_LogisticRegression_After

Second, creates upset plots comparing number of phenotypes that are significant in the primary analysis from 01a_LogisticRegression_Before, the primary analysis from 01b_LogisticRegression_After, and both analyses. Additionally, specifies which phenotypes are significant between these 3 combinations of logistic regression analyses.

#### 4a_Logit_LnLn_BH_Before

This notebook compares logistic regression analyses performed in 01a_LogisticRegression_Before

Specifically, creates ln-ln plots comparing which phenotypes are significant in the primary analysis vs:
1. crude analysis
2. social determinants of health sensitivity analysis
3. hospital utilization sensitivity analysis

#### 4b_Logit_LnLn_BH_After

This notebook compares logistic regression analyses performed in 01b_LogisticRegression_After

Specifically, creates ln-ln plots comparing which phenotypes are significant in the primary analysis vs:
1. crude analysis
2. social determinants of health sensitivity analysis
3. hospital utilization sensitivity analysis

#### 4c_Logit_LnLn_BH_Before_vs_After

This notebook compares the primary logistic regression analyses performed in 01a_LogisticRegression_Before vs 01b_LogisticRegression_After

Specifically, creates ln-ln plots comparing which phenotypes are significant in the primary analysis from 01a_LogisticRegression_Before vs the primary analysis from 01b_LogisticRegression_After

#### 05_Save_Results_Censored

Saves logistic regression analyses results from 01a_LogisticRegression_Before and 01b_LogisticRegression_After as csv files. Censors patient counts so that any phenotype where there are <= 10 patients are set to 10.

#### 6a_LogisticRegression_After_12m_Cutoff

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. These models only include patients that have at least 12 months of follow-up time and diagnoses obtained within 12 months of follow-up time. 12 months of follow-up time means that patients have 12 months of follow-up time 6 months *after* their first male infertility diagnosis or vasectomy record (i.e., 18 months after their first male infertility diagnosis or vasectomy record). *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 6b_LogisticRegression_After_24m_Cutoff 

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. These models only include patients that have at least 24 months of follow-up time and diagnoses obtained within 24 months of follow-up time. 24 months of follow-up time means that patients have 24 months of follow-up time 6 months *after* their first male infertility diagnosis or vasectomy record (i.e., 30 months after their first male infertility diagnosis or vasectomy record). *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 6c_LogisticRegression_After_36m_Cutoff 

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. These models only include patients that have at least 36 months of follow-up time and diagnoses obtained within 36 months of follow-up time. 36 months of follow-up time means that patients have 36 months of follow-up time 6 months *after* their first male infertility diagnosis or vasectomy record (i.e., 42 months after their first male infertility diagnosis or vasectomy record). *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 6d_LogisticRegression_After_48m_Cutoff 

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. These models only include patients that have at least 48 months of follow-up time and diagnoses obtained within 48 months of follow-up time. 48 months of follow-up time means that patients have 48 months of follow-up time 6 months *after* their first male infertility diagnosis or vasectomy record (i.e., 54 months after their first male infertility diagnosis or vasectomy record). *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 6e_LogisticRegression_After_60m_Cutoff 

Performs logistic regression phecode-corresponding phenotypes that first occurred greater than 6 months after diagnosis/procedure. These models only include patients that have at least 60 months of follow-up time and diagnoses obtained within 60 months of follow-up time. 60 months of follow-up time means that patients have 60 months of follow-up time 6 months *after* their first male infertility diagnosis or vasectomy record (i.e., 66 months after their first male infertility diagnosis or vasectomy record). *At least one patient each in the male infertility and vasectomy group has to be diagnosed with the phenotype in order to perform logistic regression for that phenotype.*

We ran the following logistic regression models:
1. crude: has male infertility ~ phenotype
2. primary: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care
3. social determinants of health sensitivity analysis: has male infertility ~ phentoype + age at first diagnosis/procedure + location of care + race category + ethnicity category + ADI
4. hospital utilization sensitivity analysis: male infertility ~ phentoype + age at first diagnosis/procedure + location of care + number of visits (greater than 6 months after diagnosis/procedure) + months in the EHR (greater than 6 months after diagnosis/procedure)

After running these models, the following information was included:
- patient counts for each phenotype
- significance (including Bonferroni-corrected significance and Benjamini-Hochberg significance)
- log10 odds ratio of phenotype
- -log10 p-value of phenotype
- phecode category associated phenotype (similar to ICD-10-CM categories)

For Stanford, location of care and ADI were not included.

#### 07_Save_Results_Censored_After_x_Mths

Saves logistic regression analyses results from 6a_LogisticRegression_After_12m_Cutoff, 6b_LogisticRegression_After_24m_Cutoff, 6c_LogisticRegression_After_36m_Cutoff, 6d_LogisticRegression_After_48m_Cutoff, and 6e_LogisticRegression_After_60m_Cutoff as csv files. Censors patient counts so that any phenotype where there are <= 10 patients are set to 10.

### 04_UMAP

This set of notebooks creates UMAP visualizations of patients' phenotypic profiles based on features of interest.

#### 01a_UMAP_Phe_all

Performs UMAP for all patients' phenotypes. Reduces m patients x n features to m patients x 2 UMAP coordinates that can then be used for visualization.

#### 01b_UMAP_Phe_all_before

Performs UMAP for all patients' phenotypes that were first diagnosed less than 6 months after first male infertility diagnosis or vasectomy-related procedure. 

#### 01c_UMAP_Phe_all_after

Performs UMAP for all patients' phenotypes that were first diagnosed greater than 6 months after first male infertility diagnosis or vasectomy-related procedure. 

#### 02a_UMAP_Phe_viz_mi_status_all

Visualizes UMAP for all patients' phenotypes, colored by male infertility status.

#### 02b_UMAP_Phe_viz_mi_status_before

Visualizes UMAP for all patients' phenotypes that were first diagnosed less than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by male infertility status.

#### 02c_UMAP_Phe_viz_mi_status_after

Visualizes UMAP for all patients' phenotypes that were first diagnosed greater than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by male infertility status.

#### 03_UMAP_Phe_viz_age_all

Visualizes UMAP for all patients' phenotypes, colored by age category.

#### 04_UMAP_Phe_viz_location_all

Visualizes UMAP for all patients' phenotypes, colored by UC location of care.

#### 05_UMAP_Phe_viz_race_all

Visualizes UMAP for all patients' phenotypes, colored by race category.

#### 06_UMAP_Phe_viz_ethnicity_all

Visualizes UMAP for all patients' phenotypes, colored by ethnicity category.

#### 07_UMAP_Phe_viz_adi_cat_all

Visualizes UMAP for all patients' phenotypes, colored by ADI category.

#### 08a_UMAP_Phe_viz_num_visits_all_before

Visualizes UMAP for all patients' phenotypes that were first diagnosed less than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by number of visits quintiles.

#### 08b_UMAP_Phe_viz_num_visits_all_after

Visualizes UMAP for all patients' phenotypes that were first diagnosed greater than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by number of visits quintiles.

#### 09a_UMAP_Phe_viz_months_in_EHR_all_before

Visualizes UMAP for all patients' phenotypes that were first diagnosed less than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by months in the EHR quintiles.

#### 09b_UMAP_Phe_viz_months_in_EHR_all_after

Visualizes UMAP for all patients' phenotypes that were first diagnosed greater than 6 months after first male infertility diagnosis or vasectomy-related procedure, colored by months in the EHR quintiles.

### 05_Cox_Regression

This set of notebooks contains R code for Cox Proportional Hazards Models and Kaplan-Meier plots.

#### 01_Cox_Regression

This contains R code for the following Cox Proportional Hazards Models:

coxph(Surv(time = days between male infertility diagnosis or vasectomy-related record and given diagnosis or EHR cutoff date, event = given diagnosis) 

1. primary analysis ~ has male infertility + age + location)

2. SDoH sensitivity analysis ~ has male infertility + age + location + race + ethnicity + ADI)

3. hospital utilization sensitivity analysis ~ has male infertility + age + location + number of visits + months in EHR)

#### 02_Cox_Regression_Figures

This contains R code for generating Kaplan-Meier curves.

### 06_Misc

This set of notebooks contains analyses included in supplementary data files.

#### 01_UMAP_Phe_mi_status_outliers

Identifies patients included in the outlier cluster (see UMAP visualization).

#### 02a_Obtain_ICD_SNOMED_diagnoses_outliers

Obtains SNOMED and associated ICD diagnoses for male infertility and vasectomy patients in the outlier cluster. These patients were identified in 01_UMAP_Phe_mi_status_outliers.

#### 02b_Obtain_ICD_SNOMED_diagnoses_nonoutliers

Obtains SNOMED and associated ICD diagnoses for male infertility and vasectomy patients *not* in the outlier cluster. These patients were identified in 01_UMAP_Phe_mi_status_outliers.

#### 03a_Obtain_phecodes_outliers

Obtains phecodes and phecode-corresponding phenotypes for each patients' diagnoses for patients in the outlier cluster. Adds first phenotype start date and whether it occurred less than 6 months after first diagnosis/procedure, greater than 6 months after first diagnosis/procedure, or on the same date as 6 months after first diagnosis/procedure.

#### 03b_Obtain_phecodes_nonoutliers

Obtains phecodes and phecode-corresponding phenotypes for each patients' diagnoses for patients *not* in the outlier cluster. Adds first phenotype start date and whether it occurred less than 6 months after first diagnosis/procedure, greater than 6 months after first diagnosis/procedure, or on the same date as 6 months after first diagnosis/procedure.

#### 04_Compare_outliers_vs_nonoutliers

Compares number of phecodes per patient for outliers and nonoutliers (both stratified and not stratitied by male infertility status) and runs Mann-Whitney U tests to compare. 

#### 05a_Obtain_delta_diags

Obtain time deltas between diagnoses (represented as phecode-corresponding phenotypes) and 1) first male infertiliy or vasectomy record and 2) analysis cutoff date. Time difference approximated in months.

#### 05b_Delta_quartiles_per_diagnosis

Obtains quartiles for time delta between a given diagnosis and the analysis cutoff date. Obtains deltas for diagnoses before and after the analysis cutoff date. Time difference is approximated in months.

#### 06_Patients_lost_to_followup

Number of patients lost to follow-up for each follow-up cutoff time for the after 6 month cutoff analyses. Cutoff times are 12, 24, 36, 48, or 60 months.

