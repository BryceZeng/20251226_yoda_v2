# Databricks notebook source
import os

from databricks.feature_engineering import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient

import mlflow
from mlflow.tracking import MlflowClient

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime

# COMMAND ----------

input_table_name = dbutils.widgets.get("INFERENCE_INPUT_TABLE")
env = dbutils.widgets.get("ENV")
model_name = dbutils.widgets.get("MODEL_NAME")
output_table_name = dbutils.widgets.get("OUTPUT_PREDICTION_TABLE")
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# Get datetime
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# DBTITLE 1,Load model and run inference
from predict import predict_batch

predict_batch(spark, model_uri,input_table_name, output_table_name, model_version, ts)

# COMMAND ----------

dbutils.notebook.exit(output_table_name)

# COMMAND ----------


