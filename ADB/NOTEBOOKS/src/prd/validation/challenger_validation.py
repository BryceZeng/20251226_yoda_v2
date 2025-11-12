# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

from databricks.feature_engineering import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient

import mlflow
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# config = pp.load_config(os.getenv("ENVIRONMENT", "dev"))
# print(f"Config loaded: {config}")
schema = dbutils.widgets.get("SCHEMA")
catalog = dbutils.widgets.get("CATALOG")
env = dbutils.widgets.get("ENV")
input_table_path = dbutils.widgets.get("TRAINING_DATA_PATH")
model_name = dbutils.widgets.get("MODEL_NAME")
experiment_name = dbutils.widgets.get("EXPERIMENT_NAME")
pickup_features_table = dbutils.widgets.get("PICKUP_FEATURES_TABLE")
dropoff_features_table = dbutils.widgets.get("DROP_FEATURES_TABLE")

# COMMAND ----------

if env == "prod":
  dbutils.notebook.exit(0)

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# We are interested in validating the Challenger model
model_alias = "challenger"
model_name = model_name

client = MlflowClient()
model_details = client.get_model_version_by_alias(model_name, model_alias)
model_version = int(model_details.version)

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

model_run_id = model_details.run_id
rmse_score = mlflow.get_run(model_run_id).data.metrics['test_rmse']
print(f"Current Model RMSE score: {rmse_score}")
champion_model_exists = False

try:
    # Compare the challenger RMSE score to the existing champion if it exists
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_version = int(champion_model.version)
    print(f"Champion model version: {champion_version}")
    champion_rmse = mlflow.get_run(champion_model.run_id).data.metrics['test_rmse']
    print(f'Champion RMSE score: {champion_rmse}. Challenger RMSE score: {rmse_score}.')
    metric_rmse_passed = rmse_score <= champion_rmse
    champion_model_exists = True
except:
    print(f"No Champion found. Accept the model as it's the first one.")
    metric_rmse_passed = True
    champion_model_exists = False

print(f'Model {model_name} version {model_details.version} metric_rmse_passed: {metric_rmse_passed}')
# Tag that RMSE metric check has passed
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_rmse_passed", value=metric_rmse_passed)

# COMMAND ----------

results = client.get_model_version(model_name, model_version)

if champion_model_exists:
  if results.tags["metric_rmse_passed"] == "True":
    print('register new model as Champion!')
    client.set_registered_model_alias(
      name=model_name,
      alias="Champion",
      version=model_version
    )
    print(f"Model {model_name} version {model_version} registered as Champion")

    print('register Old Model as Challenger!')
    client.set_registered_model_alias(
      name=model_name,
      alias="challenger",
      version=champion_version
    )
    print(f"Model {model_name} version {champion_version} registered as Challenger")

  else:
    raise Exception("Model not ready for promotion")
else:
  if results.tags["metric_rmse_passed"] == "True":
    print('register new model as Champion!')
    client.set_registered_model_alias(
      name=model_name,
      alias="Champion",
      version=model_version
    )
    print(f"Model {model_name} version {model_version} registered as Champion")
  else:
    raise Exception("Model not ready for promotion")
