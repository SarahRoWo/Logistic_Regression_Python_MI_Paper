# Databricks notebook source
# MAGIC %md
# MAGIC ## 20230125 Table One
# MAGIC
# MAGIC Each row in df should now correspond to one unique patient's demographic information

# COMMAND ----------

## Install dplyr (an updated version) and tableone
install.packages("dplyr")
install.packages("tableone")
install.packages("rlang")

# COMMAND ----------

## tableone package itself, as well as SparkR
library(tableone)
#library(SparkR)
library(DBI)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load demographics files of patients

# COMMAND ----------
setwd("~/Documents/Logistic_Regression_Python_Stanford/Logistic_Regression_Python_MI")
mi_pts <- read.csv("male_infertility_validation/demographics/mi_pts_only_final.csv")

vas_pts <- read.csv("male_infertility_validation/demographics/vas_pts_only_final.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add male infertility (status) column for both
# MAGIC

# COMMAND ----------

mi_pts$MaleInf <- 'male_inf'
vas_pts$MaleInf <- 'control'

# COMMAND ----------

# Uncomment this if necessary
#mi_pts <- collect(mi_pts)
#vas_pts <- collect(vas_pts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### consolidate dataframes
# MAGIC #### https://stackoverflow.com/questions/8169323/r-concatenate-two-dataframes

# COMMAND ----------

names(mi_pts)

# COMMAND ----------

names(vas_pts)

# COMMAND ----------

all_pts <- rbind(mi_pts, vas_pts)

# Check dimensions
print(dim(all_pts))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Only keep demographic data for table one
# MAGIC #### https://stackoverflow.com/questions/10085806/extracting-specific-columns-from-a-data-frame

# COMMAND ----------

all_pts <- subset(all_pts, select = -c(1))

# COMMAND ----------

names(all_pts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Table One
# MAGIC #### Main reference:
# MAGIC #### http://rstudio-pubs-static.s3.amazonaws.com/13321_da314633db924dc78986a850813a50d5.html

# COMMAND ----------

vars <- c('estimated_age', 
          'mi_or_vas_est_age',
          'gender',
          'race',
          'ethnicity',
          'num_visits_total',
          'num_visits_before',
          'num_visits_after',
          'emr_months_total',
          'emr_months_before',
          'emr_months_after',
          'MaleInf')

strata <- c('MaleInf')

# COMMAND ----------

## Create Table 1 stratified by trt (omit strata argument for overall table)
tableOne <- CreateTableOne(vars=vars, strata=strata, data=all_pts, smd=TRUE)
## Just typing the object name will invoke the print.TableOne method
## Tests are by oneway.test/t.test for continuous, chisq.test for categorical
tableOne

# COMMAND ----------

## To get SMD printed out:
## https://cran.r-project.org/web/packages/tableone/vignettes/smd.html
## And to show all levels (i.e., all levels for all variables, including binary):
## https://cran.r-project.org/web/packages/tableone/tableone.pdf
print(tableOne, smd=TRUE, showAllLevels=TRUE)

# COMMAND ----------

# Save
tableOne_save <- print(tableOne, smd=TRUE, quote=FALSE, showAllLevels=TRUE)
write.csv(tableOne_save, file = 'male_infertility_validation/tables/table_one.csv')

# COMMAND ----------

