# Databricks notebook source
# =============================================================================
# Batch Inference Pipeline for ML Model Predictions
# =============================================================================
# This notebook performs batch inference using the champion model to generate
# predictions on input data and stores results in a prediction table.
#
# Key Features:
# - Loads champion model from MLflow registry
# - Processes input data through prediction pipeline
# - Enriches predictions with metadata for tracking
# - Saves results to prediction table for downstream consumption
#
# Dependencies: predict.py module for core prediction logic
# =============================================================================

# DBTITLE 1,Import Required Libraries
# =============================================================================
# Core Libraries
# =============================================================================
import os
import uuid
from datetime import datetime

# =============================================================================
# Machine Learning Libraries
# =============================================================================
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
# =============================================================================
# Databricks & MLflow Libraries
# =============================================================================
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

# =============================================================================
# Spark Libraries for Data Processing
# =============================================================================
from pyspark.sql.functions import col, lit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("📦 All libraries imported successfully")
# =============================================================================
# Parameter Configuration
# =============================================================================
# Extract parameters passed from workflow YAML configuration
try:
    # Input/Output Configuration
    input_table_name = dbutils.widgets.get("INFERENCE_INPUT_TABLE")
    output_table_name = dbutils.widgets.get("OUTPUT_PREDICTION_TABLE")

    # Model Configuration
    env = dbutils.widgets.get("ENV")
    model_name = dbutils.widgets.get("MODEL_NAME")
    alias = "champion"  # Always use champion model for inference
    model_uri = f"models:/{model_name}@{alias}"

    # Data Schema Configuration
    id_col = dbutils.widgets.get("ID_COL")
    prediction_col = dbutils.widgets.get("PREDICTION_COL")
    project_name = dbutils.widgets.get("PROJECT_NAME")
    granularity = dbutils.widgets.get("GRANULARITY")

    print("✅ Parameter extraction successful:")
    print(f"   Input Table: {input_table_name}")
    print(f"   Output Table: {output_table_name}")
    print(f"   Model: {model_uri}")
    print(f"   Environment: {env}")

except Exception as e:
    print(f"❌ Error extracting parameters: {str(e)}")
    dbutils.notebook.exit("Failed to extract required parameters")

# =============================================================================
# Model Registry Setup
# =============================================================================
try:
    # Initialize MLflow client with Unity Catalog
    client = MlflowClient(registry_uri="databricks-uc")

    # Get the specific version of the champion model
    model_version = client.get_model_version_by_alias(model_name, alias).version

    # Generate timestamp for prediction tracking
    prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"🎯 Model configuration:")
    print(f"   Champion Model Version: {model_version}")
    print(f"   Prediction Timestamp: {prediction_timestamp}")

except Exception as e:
    error_msg = f"Failed to initialize model configuration: {str(e)}"
    print(f"❌ {error_msg}")
    dbutils.notebook.exit(error_msg)

# =============================================================================
# Core Prediction Pipeline
# =============================================================================
try:
    # Import prediction function from predict module
    from predict import predict_batch

    print(f"🚀 Starting batch inference on: {input_table_name}")

    # Execute prediction pipeline
    predictions_df = predict_batch(
        spark_session=spark,
        model_uri=model_uri,
        input_table_name=input_table_name,
        model_version=model_version,
        ts=prediction_timestamp,
        granularity=granularity,
    )

    print(f"✅ Batch inference completed successfully")
    print(f"📊 Generated {predictions_df.count()} predictions")

except Exception as e:
    error_msg = f"Batch inference failed: {str(e)}"
    print(f"❌ {error_msg}")
    dbutils.notebook.exit(error_msg)

# =============================================================================
# Output Data Preparation
# =============================================================================
try:
    print("📝 Enriching predictions with tracking metadata...")

    # Generate unique identifier for this prediction batch
    batch_uuid = uuid.uuid4().hex

    # Get prediction column data type for metadata
    prediction_type = predictions_df.schema[prediction_col].dataType.simpleString()

    # Enrich predictions with comprehensive metadata for tracking and auditing
    enriched_predictions = (
        predictions_df.withColumn("uuid", lit(batch_uuid))  # Batch identifier
        .withColumn("model_name", lit(model_name))  # Model name
        .withColumn("id", col(id_col).cast("string"))  # Record identifier
        .withColumn("model_version", lit(model_version))  # Model version
        .withColumnRenamed(
            prediction_col, "prediction"
        )  # Standardized prediction column
        .withColumn(
            "prediction", col("prediction").cast("string")
        )  # Convert to string type
        .withColumn("prediction_type", lit(prediction_type))  # Data type metadata
        .withColumn("project_name", lit(project_name))  # Project identifier
        .select(
            "uuid",
            "id",
            "model_name",
            "model_version",
            "prediction",
            "prediction_type",
            "project_name",
            "granularity",
            "timestamp",
        )
    )

    print(f"📤 Saving enriched predictions to: {output_table_name}")

    # Parse catalog, schema, and table from full table name
    table_parts = output_table_name.split(".")
    if len(table_parts) == 3:
        catalog_name, schema_name, table_name = table_parts
        # Ensure catalog and schema exist
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
        print(f"✅ Catalog and schema verified: {catalog_name}.{schema_name}")

    # Check if table exists and use appropriate write mode
    try:
        table_exists = spark.catalog.tableExists(output_table_name)
    except Exception as e:
        print(f"⚠️  Could not check table existence: {str(e)}")
        table_exists = False

    if table_exists:
        print("📋 Table exists - appending with schema evolution...")
        enriched_predictions.write.mode("append").option(
            "mergeSchema", "true"
        ).saveAsTable(output_table_name)
    else:
        print("🆕 Table doesn't exist - creating new table...")
        enriched_predictions.write.mode("overwrite").option(
            "mergeSchema", "true"
        ).saveAsTable(output_table_name)

    print("✅ Predictions saved successfully")

except Exception as e:
    error_msg = f"Failed to save predictions: {str(e)}"
    print(f"❌ {error_msg}")
    dbutils.notebook.exit(error_msg)

# =============================================================================
# Pipeline Completion
# =============================================================================
print("=" * 60)
print("🎉 BATCH INFERENCE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"📊 SUMMARY:")
print(f"   Input Table: {input_table_name}")
print(f"   Output Table: {output_table_name}")
print(f"   Model: {model_name} (v{model_version})")
print(f"   Batch UUID: {batch_uuid}")
print(f"   Environment: {env}")
print("=" * 60)

# Return output table name for downstream workflow coordination
dbutils.notebook.exit(output_table_name)
