## Runs Cox on the following diagnoses:
# 1. Abnormal spermatozoa
# 2. Arrhythmia (cardiac) NOS
# 3. Bacterial enteritis
# 4. Circulatory disease NEC
# 5. H. pylori
# 6. Osteoporosis NOS
# 7. Other disorders of metabolism
# 8. Other disorders of testis
# 9. Other disorders of the kidney and ureters
# 10. Other forms of chronic heart disease
# 11. Other tests
# 12. Persons encountering health services in circumstances related to reproduction
# 13. Testicular hypofunction

### These are the diagnoses found to be significantly associated with male infertility after the 6-month cutoff at UC for all three analyses (primary, sdoh, and hosp)

## 1. Import packages

install.packages("survminer")

library(SparkR)
library(tidyverse)
library(tidyr)
library(dplyr)
library(sparklyr)
library(lubridate)
library(survival)
library(survminer) # for customizable graphs of survival function
library(broom) # for tidy output 
library(ggplot2) # for graphing (actually loaded by survminer)

## 2. Import data

### a. demographics

mi_demo_spark <- read.df(path="path/to/male/infertility/patient/demographics",
                  source="parquet",
                  header = "true", 
                  inferSchema = "true", 
                  na.strings = "NA")

vas_only_demo_spark <- read.df(path="path/to/vasectomy/patient/demographics",
                   source="parquet",
                   header = "true", 
                   inferSchema = "true", 
                   na.strings = "NA")

# Make column specifying whether patient has mi
mi_demo_spark <- withColumn(mi_demo_spark, "has_mi", lit(1))
vas_only_demo_spark <- withColumn(vas_only_demo_spark, "has_mi", lit(0))

### b. diagnoses

mi_phe_spark <- read.df(path="path/to/male/infertility/patient/diagnoses/with/phecodes",
                  source="parquet",
                  header = "true", 
                  inferSchema = "true", 
                  na.strings = "NA")

vas_only_phe_spark <- read.df(path="path/to/vasectomy/patient/diagnoses/with/phecodes",
                   source="parquet",
                   header = "true", 
                   inferSchema = "true", 
                   na.strings = "NA")

## 2. Combine demographics and diagnoses

demo_combined_spark <- SparkR::union(mi_demo_spark, vas_only_demo_spark)
diag_combined_spark <- SparkR::union(mi_phe_spark, vas_only_phe_spark)

createOrReplaceTempView(diag_combined_spark, "diag_combined_view")
createOrReplaceTempView(demo_combined_spark, "demo_combined_view")

## 3. Specify diagnoses to test and how many patients received each diagnosis

diagnoses <- c(
  "Abnormal spermatozoa",
  "Arrhythmia (cardiac) NOS",
  "Bacterial enteritis",
  "Circulatory disease NEC",
  "H. pylori",
  "Osteoporosis NOS",
  "Other disorders of metabolism",
  "Other disorders of testis",
  "Other disorders of the kidney and ureters",
  "Other forms of chronic heart disease",
  "Other tests",
  "Persons encountering health services in circumstances related to reproduction",
  "Testicular hypofunction"
)

## 4. Functions to create Cox table for each diagnosis

process_w_phe <- function(diagnosis) {
  # This function takes in as a parameter a given diagnosis (data type string)
  # This function returns patients with specified diagnosis as well as their demographics and how many days were between their first male infertility diagnosis or vasectomy record and the specified diagnosis.

  # a. Convert diag_combined_spark and demo_combined_spark to tibbles
  diag_combined <- as_tibble(as.data.frame(diag_combined_spark))
  demo_combined <- as_tibble(as.data.frame(demo_combined_spark))
  
  # b. Select specific columns in diag_combined
  diag_combined <- diag_combined %>%
    select(person_id, phecode, phenotype, first_phe_start_date, phe_time_before, phe_time_after, phe_time_same) %>% distinct()
  
  # c. Select rows where phenotype equals diagnosis
  diag_filtered <- diag_combined %>%
    filter(phenotype == diagnosis)
  
  # d. Select person_ids found in diag_filtered in demo_combined
  demo_filtered <- demo_combined %>%
    filter(person_id %in% diag_filtered$person_id)
  
  # e. Combine selected rows via a left join
  combined_data <- diag_filtered %>%
    left_join(demo_filtered, by = "person_id")
  
  # f. Obtain difference in number of days between first_mi_or_vas_date and first_phe_start_date
  combined_data <- combined_data %>%
    mutate(days_mi_vas_phe = as.numeric(difftime(first_phe_start_date, first_mi_or_vas_date, units = "days")),
           phe_time_after_0 = ifelse(first_phe_start_date > first_mi_or_vas_date, TRUE, FALSE),
           has_phenotype = 1)
  
  # Return as tibble
  return(as_tibble(combined_data))
}

process_wo_phe <- function(result_w_phe) {
  # This function takes in as a parameter the result from process_w_phe. Here, it is named result_w_phe and contains patient information for a given diagnosis (data type tibble)
  # This function returns patients without the specified diagnosis represented in result_w_phe, as well as their demographics and how many days were between their first male infertility diagnosis or vasectomy record and the ehr cutoff date, which is June 30, 2023.

  # Convert demo_combined_spark to tibble
  demo_combined <- as_tibble(as.data.frame(demo_combined_spark))
  
  # Filter out person_ids that are in result_w_phe
  filtered_demo <- demo_combined %>%
    filter(!person_id %in% result_w_phe$person_id)
  
  # Add new columns
  ehr_cutoff_date <- as.Date("2023-06-30")
  filtered_demo <- filtered_demo %>%
    mutate(
      ehr_cutoff_date = ehr_cutoff_date,
      days_mi_vas_phe = as.numeric(difftime(ehr_cutoff_date, first_mi_or_vas_date, units = "days")),
      has_phenotype = 0
    )
}

cox_phe <- function(result_w_phe, result_wo_phe) {
  # This function takes in as a parameter the results from process_w_phe and process_wo_phe. Here, they are named result_w_phe and result_wo_phe and contain patient information with and without the given diagnosis (data type tibble)
  # This function returns patients with and without the specified diagnosis in preparation for running Cox regression.

  cat("Total number of patients who were diagnosed with phenotype: ", n_distinct(result_w_phe$person_id), "\n")
  
  # Filter result_w_phe to only include rows where phe_time_after_0 is TRUE
  filtered_w_phe <- result_w_phe %>%
    filter(phe_time_after_0 == 1)

  cat("Patients who were diagnosed with phenotype before male infertility diagnosis or vasectomy record removed...", "\n")
  
  # Select specified columns from filtered_w_phe
  selected_w_phe <- filtered_w_phe %>%
    select(person_id, days_mi_vas_phe, mi_or_vas_est_age, location_source_value, race, ethnicity, adi, num_visits_after, emr_months_after, first_mi_or_vas_date, first_phe_start_date, has_phenotype, has_mi) %>%
    rename(stop_date = first_phe_start_date)
  
  # Select specified columns from result_wo_phe
  selected_wo_phe <- result_wo_phe %>%
    select(person_id, days_mi_vas_phe, mi_or_vas_est_age, location_source_value, race, ethnicity, adi, num_visits_after, emr_months_after, first_mi_or_vas_date, ehr_cutoff_date, has_phenotype, has_mi) %>%
    rename(stop_date = ehr_cutoff_date)
  
  # Print number of distinct person_ids
  cat("Number of distinct person_ids in selected_w_phe (i.e., patients who received diagnosis after male infertility or vasectomy record):", n_distinct(selected_w_phe$person_id), "\n")
  cat("Number of distinct person_ids in selected_wo_phe:", n_distinct(selected_wo_phe$person_id), "\n")
  
  # Combine the two tibbles
  cox_table <- bind_rows(selected_w_phe, selected_wo_phe)

  cat("Table prepared for Cox...", "\n")
  
  return(cox_table)
}

convert_to_wide <- function(tidy_summary) {
  tidy_summary %>%
    pivot_longer(cols = -term, names_to = "metric", values_to = "value") %>%
    unite("term_metric", term, metric, sep = ":") %>%
    pivot_wider(names_from = term_metric, values_from = value)
}

## 5. Create for loop for each diagnosis

### a. primary

primary_results <- list()

for (diagnosis in diagnoses) {
  cat("Phenotype tested: ", diagnosis, "\n")
  result_w_phe <- process_w_phe(diagnosis)
  result_wo_phe <- process_wo_phe(result_w_phe)
  cox_table <- cox_phe(result_w_phe, result_wo_phe)
  cat("Running Cox on ", diagnosis, "...\n")
  
  cox_result <- coxph(Surv(time = days_mi_vas_phe, event = has_phenotype) ~ has_mi + mi_or_vas_est_age + location_source_value, data = cox_table)
  cox_summary <- tidy(cox_result, exponentiate = TRUE, conf.int = TRUE) %>% rename(hazard_ratio = estimate)
  
  # Convert to 1 x n tibble and add phenotype diagnosis name
  wide_cox_summary <- convert_to_wide(cox_summary)
  wide_cox_summary_phe <- add_column(wide_cox_summary, phenotype = diagnosis, .before = 1)

  # Add results to list
  cat("Adding ", diagnosis, " results to primary_results list...", "\n")
  primary_results[[diagnosis]] <- wide_cox_summary_phe
  cat("Done. \n \n")

}

cat("Completed Cox primary analysis for all diagnoses.")

# Concatenate results
primary_results_tibble <- bind_rows(primary_results)

# Add significance
primary_results_tibble_w_sig <- primary_results_tibble %>%
  mutate(significance = case_when(
    `has_mi:p.value` > 0.05 ~ "not_significant",
    `has_mi:hazard_ratio` > 1 & `has_mi:p.value` < 0.05 ~ "mi_significant",
    `has_mi:hazard_ratio` < 1 & `has_mi:p.value` < 0.05 ~ "con_significant"
  ))

display(primary_results_tibble_w_sig)

# Convert to SparkDataFrame
primary_results_df <- as.data.frame(primary_results_tibble_w_sig)
primary_results_spark_df <- createDataFrame(primary_results_df)

# Save primary_results_spark_df to workspace
write.df(primary_results_spark_df, path = "path/to/cox/primary_results", source = "parquet", mode = "overwrite")

### b. sdoh

sdoh_results <- list()

for (diagnosis in diagnoses) {
  cat("Phenotype tested: ", diagnosis, "\n")
  result_w_phe <- process_w_phe(diagnosis)
  result_wo_phe <- process_wo_phe(result_w_phe)
  cox_table <- cox_phe(result_w_phe, result_wo_phe)
  cat("Running Cox on ", diagnosis, "...\n")
  
  cox_result <- coxph(Surv(time = days_mi_vas_phe, event = has_phenotype) ~ has_mi + mi_or_vas_est_age + location_source_value + race + ethnicity + adi, data = cox_table)
  cox_summary <- tidy(cox_result, exponentiate = TRUE, conf.int = TRUE) %>% rename(hazard_ratio = estimate)
  
  # Convert to 1 x n tibble and add phenotype diagnosis name
  wide_cox_summary <- convert_to_wide(cox_summary)
  wide_cox_summary_phe <- add_column(wide_cox_summary, phenotype = diagnosis, .before = 1)

  # Add results to list
  cat("Adding ", diagnosis, " results to sdoh_results list...", "\n")
  sdoh_results[[diagnosis]] <- wide_cox_summary_phe
  cat("Done. \n \n")

}

cat("Completed Cox sdoh analysis for all diagnoses.")

# Concatenate results
sdoh_results_tibble <- bind_rows(sdoh_results)

# Add significance
sdoh_results_tibble_w_sig <- sdoh_results_tibble %>%
  mutate(significance = case_when(
    `has_mi:p.value` > 0.05 ~ "not_significant",
    `has_mi:hazard_ratio` > 1 & `has_mi:p.value` < 0.05 ~ "mi_significant",
    `has_mi:hazard_ratio` < 1 & `has_mi:p.value` < 0.05 ~ "con_significant"
  ))

display(sdoh_results_tibble_w_sig)

# Convert to SparkDataFrame
sdoh_results_df <- as.data.frame(sdoh_results_tibble_w_sig)
sdoh_results_spark_df <- createDataFrame(sdoh_results_df)

# Save sdoh_results_spark_df to workspace
write.df(sdoh_results_spark_df, path = "s3://uchdw-501868445017-us-west-2-prod-databricks-user-files/ucsf_sarah_woldemariam/male_infertility/cox/sdoh_results", source = "parquet", mode = "overwrite")

### c. hosp

hosp_results <- list()

for (diagnosis in diagnoses) {
  cat("Phenotype tested: ", diagnosis, "\n")
  result_w_phe <- process_w_phe(diagnosis)
  result_wo_phe <- process_wo_phe(result_w_phe)
  cox_table <- cox_phe(result_w_phe, result_wo_phe)
  cat("Running Cox on ", diagnosis, "...\n")
  
  cox_result <- coxph(Surv(time = days_mi_vas_phe, event = has_phenotype) ~ has_mi + mi_or_vas_est_age + location_source_value + num_visits_after + emr_months_after, data = cox_table)
  cox_summary <- tidy(cox_result, exponentiate = TRUE, conf.int = TRUE) %>% rename(hazard_ratio = estimate)
  
  # Convert to 1 x n tibble and add phenotype diagnosis name
  wide_cox_summary <- convert_to_wide(cox_summary)
  wide_cox_summary_phe <- add_column(wide_cox_summary, phenotype = diagnosis, .before = 1)

  # Add results to list
  cat("Adding ", diagnosis, " results to hosp_results list...", "\n")
  hosp_results[[diagnosis]] <- wide_cox_summary_phe
  cat("Done. \n \n")

}

cat("Completed Cox hosp analysis for all diagnoses.")

# Concatenate results
hosp_results_tibble <- bind_rows(hosp_results)

# Add significance
hosp_results_tibble_w_sig <- hosp_results_tibble %>%
  mutate(significance = case_when(
    `has_mi:p.value` > 0.05 ~ "not_significant",
    `has_mi:hazard_ratio` > 1 & `has_mi:p.value` < 0.05 ~ "mi_significant",
    `has_mi:hazard_ratio` < 1 & `has_mi:p.value` < 0.05 ~ "con_significant"
  ))

display(hosp_results_tibble_w_sig)

# Convert to SparkDataFrame
hosp_results_df <- as.data.frame(hosp_results_tibble_w_sig)
hosp_results_spark_df <- createDataFrame(hosp_results_df)

# Save hosp_results_spark_df to workspace
write.df(hosp_results_spark_df, path = "path/to/cox/hosp_results", source = "parquet", mode = "overwrite")

## 7. Save as csv files

# COMMAND ----------

# primary
write.df(primary_results_spark_df,"path/to/cox/csv/primary_results", "csv", header=TRUE, mode = "overwrite")

# sdoh
write.df(sdoh_results_spark_df,"path/to/cox/csv/sdoh_results", "csv", header=TRUE, mode = "overwrite")

# hosp
write.df(hosp_results_spark_df,"path/to/cox/csv/hosp_results", "csv", header=TRUE, mode = "overwrite")
