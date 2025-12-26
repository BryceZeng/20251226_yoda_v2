# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.tracking.client import MlflowClient

# COMMAND ----------


schema = dbutils.widgets.get("SCHEMA")
catalog = dbutils.widgets.get("CATALOG")
env = dbutils.widgets.get("ENV")
model_name = dbutils.widgets.get("MODEL_NAME")
experiment_name = dbutils.widgets.get("EXPERIMENT_NAME")
# pickup_features_table = dbutils.widgets.get("PICKUP_FEATURES_TABLE")
# dropoff_features_table = dbutils.widgets.get("DROP_FEATURES_TABLE")

# Construct the full 3-level Unity Catalog model name
full_model_name = f"{catalog}.{schema}.{model_name}"

print(f"🔍 Configuration:")
print(f"   Catalog: {catalog}")
print(f"   Schema: {schema}")
print(f"   Model Name: {model_name}")
print(f"   Full Model Name: {full_model_name}")

# COMMAND ----------

if env == "prod":
    dbutils.notebook.exit(0)

# COMMAND ----------

# Skip strict validation in dev environment
if env.lower() == "dev":
    print("⚠️  DEV ENVIRONMENT: Skipping strict performance validation")
    print("   All models will be promoted in dev for testing purposes")

    client = MlflowClient(registry_uri="databricks-uc")
    mlflow.set_registry_uri('databricks-uc')

    # Get the challenger model details
    model_alias = "challenger"
    model_details = client.get_model_version_by_alias(full_model_name, model_alias)
    model_version = int(model_details.version)

    # Simply promote to Champion without validation
    print(f"🚀 Promoting model {full_model_name} version {model_version} to Champion")
    client.set_registered_model_alias(
        name=full_model_name, alias="Champion", version=model_version
    )
    print(f"✅ Model promoted successfully in dev environment")
    dbutils.notebook.exit(0)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# We are interested in validating the Challenger model
model_alias = "challenger"

client = MlflowClient()
model_details = client.get_model_version_by_alias(full_model_name, model_alias)
model_version = int(model_details.version)

print(
    f"Validating {model_alias} model for {full_model_name} on model version {model_version}"
)

# COMMAND ----------

model_run_id = model_details.run_id
rmse_score = mlflow.get_run(model_run_id).data.metrics["test_rmse"]
print(f"Current Challenger Model RMSE score: {rmse_score}")
print(f"Challenger model version: {model_version}")
champion_model_exists = False

try:
    # Compare the challenger RMSE score to the existing champion if it exists
    print("🔍 Checking for existing Champion model...")
    champion_model = client.get_model_version_by_alias(full_model_name, "Champion")
    champion_version = int(champion_model.version)
    print(f"✅ Found Champion model version: {champion_version}")
    champion_rmse = mlflow.get_run(champion_model.run_id).data.metrics["test_rmse"]
    print(f"📊 Champion RMSE: {champion_rmse} vs Challenger RMSE: {rmse_score}")
    metric_rmse_passed = rmse_score <= champion_rmse
    champion_model_exists = True

    if metric_rmse_passed:
        print(
            f"✅ Challenger model BETTER - RMSE improved by {champion_rmse - rmse_score:.6f}"
        )
    else:
        print(
            f"❌ Challenger model WORSE - RMSE increased by {rmse_score - champion_rmse:.6f}"
        )

except Exception as e:
    print(f"⚠️  No Champion found (expected for first model): {e}")
    print("   Accepting challenger as the first champion model.")
    metric_rmse_passed = True
    champion_model_exists = False

print(f"🏁 VALIDATION RESULT:")
print(f"   Model: {full_model_name} version {model_details.version}")
print(f"   RMSE Check Passed: {metric_rmse_passed}")
print(f"   Champion Exists: {champion_model_exists}")
# Tag that RMSE metric check has passed - ensure string value for consistent comparison
client.set_model_version_tag(
    name=full_model_name,
    version=model_details.version,
    key="metric_rmse_passed",
    value=str(metric_rmse_passed),
)

# COMMAND ----------

import time

# Small delay to ensure tag is persisted
time.sleep(2)

results = client.get_model_version(full_model_name, model_version)

# Debug: Print all available information
print(f"Debug - metric_rmse_passed value before tagging: {metric_rmse_passed}")
print(f"Debug - All tags: {results.tags}")
print(
    f"Debug - metric_rmse_passed tag value: '{results.tags.get('metric_rmse_passed')}' (type: {type(results.tags.get('metric_rmse_passed'))})"
)
print(f"Debug - champion_model_exists: {champion_model_exists}")

if champion_model_exists:
    print("🏆 CHAMPION MODEL EXISTS - COMPARISON MODE")
    # More robust check for metric_rmse_passed tag
    tag_value = results.tags.get("metric_rmse_passed")
    passed = tag_value == "True" or tag_value == True or tag_value == "true"

    print(f"   Tag value: '{tag_value}' -> Validation passed: {passed}")

    if passed:
        print("🚀 PROMOTING challenger to Champion!")
        client.set_registered_model_alias(
            name=full_model_name, alias="Champion", version=model_version
        )
        print(
            f"✅ Model {full_model_name} version {model_version} registered as Champion"
        )

        print("🔄 Moving old Champion to challenger...")
        client.set_registered_model_alias(
            name=full_model_name, alias="challenger", version=champion_version
        )
        print(
            f"✅ Model {full_model_name} version {champion_version} registered as challenger"
        )

    else:
        print(
            "❌ VALIDATION FAILED - Challenger model performance is worse than Champion"
        )
        print(f"   Challenger RMSE: {rmse_score}")
        print(f"   Champion RMSE: {champion_rmse}")
        print(f"   Tag value: '{results.tags.get('metric_rmse_passed')}'")
        print(
            "   Recommendation: Check model training parameters or feature engineering"
        )
        raise Exception("Model not ready for promotion - performance degraded")
else:
    print("🎯 FIRST MODEL - NO CHAMPION EXISTS")
    # More robust check for metric_rmse_passed tag
    tag_value = results.tags.get("metric_rmse_passed")
    passed = tag_value == "True" or tag_value == True or tag_value == "true"

    print(f"   Tag value: '{tag_value}' -> Validation passed: {passed}")

    if passed:
        print("🚀 PROMOTING first model to Champion!")
        client.set_registered_model_alias(
            name=full_model_name, alias="Champion", version=model_version
        )
        print(
            f"✅ Model {full_model_name} version {model_version} registered as Champion"
        )
    else:
        print("❌ UNEXPECTED: First model failed basic validation")
        print(f"   RMSE Score: {rmse_score}")
        print(f"   Tag value: '{results.tags.get('metric_rmse_passed')}'")
        print("   This should not happen for the first model")
        raise Exception("Model not ready for promotion - unexpected validation failure")
