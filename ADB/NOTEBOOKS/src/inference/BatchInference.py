# Databricks notebook source
# MAGIC %pip install -qq databricks-feature-engineering
# MAGIC %pip install -qq lightgbm
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

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
id_col = dbutils.widgets.get("ID_COL")
prediction_col = dbutils.widgets.get("PREDICTION_COL")

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# Get datetime
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# DBTITLE 1,Load model and run inference
from predict import predict_batch

predict_df = predict_batch(spark, model_uri,input_table_name, model_version, ts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Prediction table

# COMMAND ----------

from pyspark.sql.functions import lit,col
import uuid
uuid = uuid.uuid4().hex
df = predict_df.withColumn("uuid", lit(uuid)) \
  .withColumn("model_name",lit(model_name)) \
  .withColumn("id",col(id_col).cast("string")) \
  .withColumn("model_version",lit(model_version)) \
  .withColumnRenamed(prediction_col,"prediction") \
  .select("uuid","id","model_name","model_version","prediction","timestamp")

df.write.mode("append").saveAsTable(output_table_name)


# COMMAND ----------

dbutils.notebook.exit(output_table_name)

# COMMAND ----------


