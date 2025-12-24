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
    
    # HARDCODED OUTPUT TABLE: All predictions from ALL projects write to a common location
    # This enables cross-project analytics and centralized monitoring in feature_store schema
    # Format: ai_engineering.feature_store.predictions (NOT project-specific)
    output_table_name = "ai_engineering.feature_store.predictions"

    # Model Configuration
    env = dbutils.widgets.get("ENV")
    model_name_base = dbutils.widgets.get("MODEL_NAME")
    # Construct 3-level Unity Catalog model name: catalog.schema.model
    model_name = f"ai_engineering.prj_yoda.{model_name_base}"
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
    print(f"   Model Name (3-level): {model_name}")
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

    # Check if model exists
    try:
        model_info = client.get_registered_model(model_name)
        print(f"✅ Model found: {model_name}")
    except Exception as model_err:
        error_msg = f"Model '{model_name}' not found in registry: {str(model_err)}"
        print(f"❌ {error_msg}")
        print(
            "💡 Tip: Ensure the training workflow has completed and registered a model"
        )
        dbutils.notebook.exit(error_msg)

    # Get the specific version of the champion model
    try:
        model_version = client.get_model_version_by_alias(model_name, alias).version
        print(f"✅ Found champion model alias")
    except Exception as alias_err:
        error_msg = (
            f"No '{alias}' alias found for model '{model_name}': {str(alias_err)}"
        )
        print(f"❌ {error_msg}")
        print(
            f"💡 Tip: Run the model training workflow first or set the champion alias manually"
        )
        dbutils.notebook.exit(error_msg)

    # Generate timestamp for prediction tracking
    prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"🎯 Model configuration:")
    print(f"   Champion Model Version: {model_version}")
    print(f"   Model URI: {model_uri}")
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

    prediction_count = predictions_df.count()
    print(f"✅ Batch inference completed successfully")
    print(f"📊 Generated {prediction_count} predictions")

    if prediction_count == 0:
        print("⚠️  WARNING: No predictions were generated!")
        print(f"   Input table '{input_table_name}' may be empty or have no valid rows")
        print("   Exiting without writing to predictions table")
        dbutils.notebook.exit("No predictions generated - input data empty")

except Exception as e:
    error_msg = f"Batch inference failed: {str(e)}"
    print(f"❌ {error_msg}")
    dbutils.notebook.exit(error_msg)

# =============================================================================
# Output Data Preparation
# =============================================================================
try:
    print("📝 Enriching predictions with tracking metadata...")

    # Debug: Show what columns we have
    print(f"📋 Available columns: {predictions_df.columns}")
    print(f"🔍 Looking for column: '{prediction_col}'")

    # Generate unique identifier for this prediction batch
    batch_uuid = uuid.uuid4().hex

    # Validate prediction column exists
    if prediction_col not in predictions_df.columns:
        available_cols = ", ".join(predictions_df.columns)
        error_msg = f"Prediction column '{prediction_col}' not found in DataFrame. Available columns: {available_cols}"
        print(f"❌ {error_msg}")
        dbutils.notebook.exit(error_msg)

    # Get prediction column data type for metadata
    prediction_type = predictions_df.schema[prediction_col].dataType.simpleString()

    # Validate ID column exists
    if id_col not in predictions_df.columns:
        available_cols = ", ".join(predictions_df.columns)
        error_msg = f"ID column '{id_col}' not found in DataFrame. Available columns: {available_cols}"
        print(f"❌ {error_msg}")
        dbutils.notebook.exit(error_msg)

    print(f"✅ Validated required columns: '{id_col}' and '{prediction_col}'")

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

        # Set current catalog for Unity Catalog
        spark.sql(f"USE CATALOG {catalog_name}")
        print(f"✅ Using catalog: {catalog_name}")

        # Ensure catalog and schema exist
        try:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
            print(f"✅ Catalog and schema verified: {catalog_name}.{schema_name}")
        except Exception as e:
            print(f"⚠️  Catalog/schema creation warning: {str(e)}")
    else:
        error_msg = f"Invalid table name format: '{output_table_name}'. Expected format: catalog.schema.table"
        print(f"❌ {error_msg}")
        dbutils.notebook.exit(error_msg)

    # Display sample of data to be written
    print("📋 Sample of enriched predictions:")
    enriched_predictions.show(5, truncate=False)

    # Check if table exists and use appropriate write mode
    try:
        table_exists = spark.catalog.tableExists(output_table_name)
    except Exception as e:
        print(f"⚠️  Could not check table existence: {str(e)}")
        table_exists = False

    if table_exists:
        print("📋 Table exists - performing MERGE to prevent duplicates...")

        from delta.tables import DeltaTable

        # Get the Delta table
        delta_table = DeltaTable.forName(spark, output_table_name)

        # Perform merge operation (upsert based on ID)
        # - WHEN MATCHED: Update the existing record with new prediction
        # - WHEN NOT MATCHED: Insert the new record
        print(f"🔄 Merging predictions based on ID column to avoid duplicates...")
        delta_table.alias("target").merge(
            enriched_predictions.alias("source"), "target.id = source.id"
        ).whenMatchedUpdate(
            set={
                "uuid": col("source.uuid"),
                "model_name": col("source.model_name"),
                "model_version": col("source.model_version"),
                "prediction": col("source.prediction"),
                "prediction_type": col("source.prediction_type"),
                "project_name": col("source.project_name"),
                "granularity": col("source.granularity"),
                "timestamp": col("source.timestamp"),
            }
        ).whenNotMatchedInsertAll().execute()

        print("✅ MERGE operation completed - no duplicate IDs created")

    else:
        print("🆕 Table doesn't exist - creating new table...")
        print(
            f"💾 Writing {enriched_predictions.count()} records in 'overwrite' mode..."
        )
        enriched_predictions.write.format("delta").mode("overwrite").option(
            "mergeSchema", "true"
        ).option("overwriteSchema", "true").saveAsTable(output_table_name)
        print("✅ Predictions table created successfully")

    # Verify write by reading back
    result_count = spark.table(output_table_name).count()
    print(f"📊 Verification: Table now contains {result_count} total records")

    # Check for duplicates
    duplicate_check = spark.sql(
        f"""
        SELECT id, COUNT(*) as count
        FROM {output_table_name}
        GROUP BY id
        HAVING COUNT(*) > 1
    """
    )

    duplicate_count = duplicate_check.count()
    if duplicate_count > 0:
        print(f"⚠️  WARNING: Found {duplicate_count} duplicate IDs!")
        duplicate_check.show(10)
    else:
        print(f"✅ No duplicate IDs found - data integrity confirmed")

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
print(f"   Records Processed: {prediction_count}")
print("=" * 60)

# Return output table name for downstream workflow coordination
dbutils.notebook.exit(output_table_name)
