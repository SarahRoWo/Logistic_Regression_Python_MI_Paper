# 20250514

## Running Cox and making Kaplan-Meier curves for the following diagnoses:
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

process_w_phe_years <- function(diagnosis) {
  # This function takes in as a parameter a given diagnosis (data type string)
  # This function returns patients with specified diagnosis as well as their demographics and how many years were between their first male infertility diagnosis or vasectomy record and the specified diagnosis.

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
  
  # f. Obtain difference in number of years between first_mi_or_vas_date and first_phe_start_date 
  combined_data <- combined_data %>%
    mutate(years_mi_vas_phe = time_length(interval(first_mi_or_vas_date, first_phe_start_date), unit = "year"),
           #years_mi_vas_phe = as.numeric(difftime(first_phe_start_date, first_mi_or_vas_date, units = "years")),
           phe_time_after_0 = ifelse(first_phe_start_date > first_mi_or_vas_date, TRUE, FALSE),
           has_phenotype = 1)
  
  # Return as tibble
  return(as_tibble(combined_data))
}

process_wo_phe_years <- function(result_w_phe) {
  # This function takes in as a parameter the result from process_w_phe. Here, it is named result_w_phe and contains patient information for a given diagnosis (data type tibble)
  # This function returns patients without the specified diagnosis represented in result_w_phe, as well as their demographics and how many years were between their first male infertility diagnosis or vasectomy record and the ehr cutoff date, which is June 30, 2023.

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
      years_mi_vas_phe = time_length(interval(first_mi_or_vas_date, ehr_cutoff_date), unit = "year"),
      #years_mi_vas_phe = as.numeric(difftime(ehr_cutoff_date, first_mi_or_vas_date, units = "years")),
      has_phenotype = 0
    )
}

km_phe <- function(result_w_phe, result_wo_phe) {
  # This function takes in as a parameter the results from process_w_phe and process_wo_phe. Here, they are named result_w_phe and result_wo_phe and contain patient information with and without the given diagnosis (data type tibble)
  # This function returns patients with and without the specified diagnosis in preparation for generating Kaplan Meier curve.

  cat("Total number of patients who were diagnosed with phenotype: ", n_distinct(result_w_phe$person_id), "\n")
  
  # Filter result_w_phe to only include rows where phe_time_after_0 is TRUE
  filtered_w_phe <- result_w_phe %>%
    filter(phe_time_after_0 == 1)

  cat("Patients who were diagnosed with phenotype before male infertility diagnosis or vasectomy record removed...", "\n")
  
  # Select specified columns from filtered_w_phe
  selected_w_phe <- filtered_w_phe %>%
    select(person_id, years_mi_vas_phe, mi_or_vas_est_age, location_source_value, race, ethnicity, adi, num_visits_after, emr_months_after, first_mi_or_vas_date, first_phe_start_date, has_phenotype, has_mi) %>%
    rename(stop_date = first_phe_start_date)
  
  # Select specified columns from result_wo_phe
  selected_wo_phe <- result_wo_phe %>%
    select(person_id, years_mi_vas_phe, mi_or_vas_est_age, location_source_value, race, ethnicity, adi, num_visits_after, emr_months_after, first_mi_or_vas_date, ehr_cutoff_date, has_phenotype, has_mi) %>%
    rename(stop_date = ehr_cutoff_date)
  
  # Print number of distinct person_ids
  cat("Number of distinct person_ids in selected_w_phe (i.e., patients who received diagnosis after male infertility or vasectomy record):", n_distinct(selected_w_phe$person_id), "\n")
  cat("Number of distinct person_ids in selected_wo_phe:", n_distinct(selected_wo_phe$person_id), "\n")
  
  # Combine the two tibbles
  km_table <- bind_rows(selected_w_phe, selected_wo_phe)

  cat("Table prepared for KM survival curve...", "\n")
  
  return(km_table)
}

convert_to_wide <- function(tidy_summary) {
  tidy_summary %>%
    pivot_longer(cols = -term, names_to = "metric", values_to = "value") %>%
    unite("term_metric", term, metric, sep = ":") %>%
    pivot_wider(names_from = term_metric, values_from = value)
}

wrap_title <- function(title, width = 40) {
  paste(strwrap(title, width = width), collapse = "\n")
}

## 5. Create KM survival plot for each diagnosis

# Collect plots
plots <- list()

for (diagnosis in diagnoses) {
  cat("Phenotype tested: ", diagnosis, "\n")
  result_w_phe <- process_w_phe_years(diagnosis)
  result_wo_phe <- process_wo_phe_years(result_w_phe)
  km_table <- km_phe(result_w_phe, result_wo_phe)
  cat("Generating KM survival curve for ", diagnosis, "...\n")
  
  KM.has_mi <- survfit(Surv(time = years_mi_vas_phe, event = has_phenotype) ~ has_mi, data=km_table)

  surv_plot <- ggsurvplot(KM.has_mi, 
                          ylim = c(0.80, 1),
                          xlab = "time (years)", 
                          ylab = "probability without\ndiagnosis", 
                          title = wrap_title(diagnosis, width = 20),
                          font.main = c(35, "bold"),
                          font.x = c(33),
                          font.y = c(33),
                          font.tickslab = c(31),
                          font.legend = c(26),
                          palette = c("#3499CC", "#EC7CB4"),
                          conf.int = TRUE)

  # Save to workspace
  filename <- paste0(diagnosis, " bf.pdf")  
  ggsave(filename, surv_plot$plot)

  plots[[length(plots) + 1]] <- surv_plot$plot

}

cat("Completed KM survival curves for all diagnoses.")
