# Databricks notebook source
import os

# COMMAND ----------

if not os.path.isdir('male_infertility_validation'):
  os.mkdir('male_infertility_validation')
if not os.path.isdir('male_infertility_validation/demographics'):
  os.mkdir('male_infertility_validation/demographics')
if not os.path.isdir('male_infertility_validation/diagnoses'):
  os.mkdir('male_infertility_validation/diagnoses')
if not os.path.isdir('male_infertility_validation/tables'):
  os.mkdir('male_infertility_validation/tables')
if not os.path.isdir('male_infertility_validation/concepts'):
  os.mkdir('male_infertility_validation/tables/concepts')
if not os.path.isdir('male_infertility_validation/raw_data'):
  os.mkdir('male_infertility_validation/raw_data')
if not os.path.isdir('male_infertility_validation/phecodes'):
  os.mkdir('male_infertility_validation/phecodes')
if not os.path.isdir('male_infertility_validation/tables/umap'):
  os.mkdir('male_infertility_validation/tables/umap')
if not os.path.isdir('male_infertility_validation/tables/umap/dunns_test'):
  os.mkdir('male_infertility_validation/tables/umap/dunns_test')
if not os.path.isdir('male_infertility_validation/tables/logit_results'):
  os.mkdir('male_infertility_validation/tables/logit_results')
if not os.path.isdir('male_infertility_validation/tables/logit_results/before'):
  os.mkdir('male_infertility_validation/tables/logit_results/before')
if not os.path.isdir('male_infertility_validation/tables/logit_results/after'):
  os.mkdir('male_infertility_validation/tables/logit_results/after')
if not os.path.isdir('male_infertility_validation/tables/logit_results/before_censored'):
  os.mkdir('male_infertility_validation/tables/logit_results/before_censored')
if not os.path.isdir('male_infertility_validation/tables/logit_results/after_censored'):
  os.mkdir('male_infertility_validation/tables/logit_results/after_censored')
if not os.path.isdir('male_infertility_validation/tables/upset'):
  os.mkdir('male_infertility_validation/tables/upset')
if not os.path.isdir('male_infertility_validation/figures'):
  os.mkdir('male_infertility_validation/figures')
if not os.path.isdir('male_infertility_validation/figures/umap'):
  os.mkdir('male_infertility_validation/figures/umap')
if not os.path.isdir('male_infertility_validation/figures/umap/cluster'):
  os.mkdir('male_infertility_validation/figures/umap/cluster')
if not os.path.isdir('male_infertility_validation/figures/umap/violin'):
  os.mkdir('male_infertility_validation/figures/umap/violin')
if not os.path.isdir('male_infertility_validation/figures/logit'):
  os.mkdir('male_infertility_validation/figures/logit')
if not os.path.isdir('male_infertility_validation/figures/logit/volcano'):
  os.mkdir('male_infertility_validation/figures/logit/volcano')
if not os.path.isdir('male_infertility_validation/figures/logit/manhattan'):
  os.mkdir('male_infertility_validation/figures/logit/manhattan')
if not os.path.isdir('male_infertility_validation/figures/logit/loglog'):
  os.mkdir('male_infertility_validation/figures/logit/loglog')
if not os.path.isdir('male_infertility_validation/figures/logit/upset'):
  os.mkdir('male_infertility_validation/figures/logit/upset')

# COMMAND ----------

