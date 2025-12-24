# =============================================================================
# Feature Engineering Module for Batch Inference
# =============================================================================
# This module provides feature engineering and prediction functionality for
# batch inference pipelines. It includes the same feature engineering logic
# used during training to ensure consistency.
#
# Key Functions:
# - engineer_features: Applies feature engineering transformations
# - predict_batch: Main prediction pipeline with data loading and enrichment
# =============================================================================

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import mlflow
import pandas as pd
import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import lit, struct, to_timestamp
from pyspark.sql.types import IntegerType


def engineer_features(df):
    """
    Engineer features from raw data with null handling.

    This applies the same feature engineering logic used during training
    to ensure consistency between training and inference.

    Args:
        df: Pandas DataFrame with raw customer data

    Returns:
        Pandas DataFrame with engineered features
    """
    df = df.copy()

    # 1. Count policy_number (assuming it's a list/array column)
    if "policy_number" in df.columns:
        df["policy_count"] = df["policy_number"].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) and x is not None else 0
        )

    # 2. Age calculation: trans_yyyymm - date_of_birth
    if "trans_yyyymm" in df.columns and "date_of_birth" in df.columns:
        df["trans_yyyymm_dt"] = pd.to_datetime(
            df["trans_yyyymm"] + "-01", errors="coerce"
        )
        df["date_of_birth_dt"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

        df["age"] = (
            (df["trans_yyyymm_dt"] - df["date_of_birth_dt"]).dt.days / 365.25
        ).fillna(-1)
        df["age"] = df["age"].clip(lower=0, upper=120)

    # 3. Months since communication consent: trans_yyyymm - communication_consent_date
    if "trans_yyyymm" in df.columns and "communication_consent_date" in df.columns:
        df["communication_consent_date_dt"] = pd.to_datetime(
            df["communication_consent_date"], errors="coerce"
        )

        df["months_since_consent"] = (
            (
                df["trans_yyyymm_dt"].dt.year
                - df["communication_consent_date_dt"].dt.year
            )
            * 12
            + (
                df["trans_yyyymm_dt"].dt.month
                - df["communication_consent_date_dt"].dt.month
            )
        ).fillna(-1)

    # 4. Handle categorical nulls
    categorical_cols = [
        "gender",
        "marital_status",
        "segment_description",
        "microsegment",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    # 5. Handle numerical nulls
    numerical_cols = ["salary", "bmi", "ctp_value", "upgrader_shortfall"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    # 6. Handle binary indicators
    binary_cols = ["is_smoker", "hazardous_lifestyle_ind", "communication_consent"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def predict_batch(
    spark_session,
    model_uri: str,
    input_table_name: str,
    model_version: str,
    ts: str,
    granularity: str = "party",
):
    """
    Executes batch prediction on input data with feature engineering.

    This function loads data from the input table and applies feature engineering
    transformations before generating predictions. It intelligently determines the
    date range to process based on existing predictions.

    Args:
        spark_session: Active Spark session for data processing
        model_uri: MLflow model URI (e.g., "models:/model_name@alias")
        input_table_name: Full table name to read input data from (catalog.schema.table)
        model_version: Version identifier of the model being used
        ts: Timestamp string for prediction metadata
        granularity: Granularity level for predictions (e.g., party, claims, agent)

    Returns:
        PySpark DataFrame with predictions and metadata columns

    Raises:
        Exception: If model loading or prediction fails
    """
    print("🚀 Initializing batch prediction pipeline...")

    # Configure MLflow registry
    mlflow.set_registry_uri("databricks-uc")

    # =============================================================================
    # Check Last Prediction Date from predictions table
    # =============================================================================
    predictions_table = "ai_engineering.feature_store.predictions"
    print(f"🔍 Checking last prediction date from {predictions_table}...")

    try:
        # Check if predictions table exists
        table_exists = spark_session.catalog.tableExists(predictions_table)

        if table_exists:
            # Get the last prediction date from the table
            last_prediction_query = f"""
                SELECT MAX(DATE(timestamp)) as last_prediction_date
                FROM {predictions_table}
            """
            last_prediction_result = spark_session.sql(last_prediction_query).collect()
            last_prediction_date = last_prediction_result[0]["last_prediction_date"]

            if last_prediction_date:
                print(f"✅ Found last prediction date: {last_prediction_date}")
                # Calculate the next date to start predictions from (add 3 months)
                last_date = datetime.strptime(str(last_prediction_date), "%Y-%m-%d")
                start_date = last_date + relativedelta(months=3)
                start_date_str = start_date.strftime("%Y-%m-%d")
                print(
                    f"📅 Starting predictions from: {start_date_str} (3 months after last prediction)"
                )
                has_existing_predictions = True
            else:
                print(
                    "⚠️  Predictions table exists but is empty - will create from scratch"
                )
                start_date_str = None
                has_existing_predictions = False
        else:
            print("⚠️  Predictions table does not exist - will create from scratch")
            start_date_str = None
            has_existing_predictions = False

    except Exception as e:
        print(f"⚠️  Could not check predictions table: {str(e)}")
        print("   Will process all available data")
        start_date_str = None
        has_existing_predictions = False

    # =============================================================================
    # Data Loading - Read from input table with date filtering
    # =============================================================================
    print(f"📊 Loading data from input table: {input_table_name}...")

    try:
        # Build query based on whether we have existing predictions
        # This follows the same pattern as TrainWithFeatureStore.py which uses b_snapshot_month
        if has_existing_predictions and start_date_str:
            # Only get data from the start_date onwards
            today = datetime.now().strftime("%Y-%m-%d")

            query = f"""
                SELECT * FROM {input_table_name}
                WHERE snapshot_date >= '{start_date_str}'
                  AND snapshot_date <= '{today}'
            """
            print(f"📆 Filtering data from {start_date_str} to {today}")
        else:
            # Get all available data if no existing predictions
            query = f"SELECT * FROM {input_table_name}"
            print(f"📆 Processing all available data (no existing predictions)")

        base_df = spark_session.sql(query)
        row_count = base_df.count()
        col_count = len(base_df.columns)
        print(f"✅ Data loaded: {row_count} rows, {col_count} columns")

        if row_count == 0:
            print(f"⚠️  WARNING: No new data to process!")
            print(
                f"   Either input table '{input_table_name}' is empty or all data has been processed"
            )
            return spark_session.createDataFrame([], base_df.schema)

    except Exception as e:
        error_msg = f"Failed to load data from '{input_table_name}': {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    # =============================================================================
    # Feature Engineering - Apply same transformations as training
    # =============================================================================
    print("🔄 Applying feature engineering...")

    try:
        # Convert to Pandas for feature engineering
        data_pdf = base_df.toPandas()
        print(f"✅ Converted to Pandas: {len(data_pdf)} rows")

        # Apply feature engineering
        data_pdf = engineer_features(data_pdf)
        print(f"✅ Feature engineering completed")

        # Define feature columns (same as training)
        feature_cols = [
            "gender",
            "marital_status",
            "salary",
            "is_smoker",
            "policy_count",
            "bmi",
            "hazardous_lifestyle_ind",
            "communication_consent",
            "segment_description",
            "ctp_value",
            "upgrader_shortfall",
            "microsegment",
            "age",
            "months_since_consent",
        ]

        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in data_pdf.columns]
        print(f"📊 Using {len(feature_cols)} features for prediction")

        X = data_pdf[feature_cols]

    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        raise Exception(f"Feature engineering failed: {str(e)}")

    # =============================================================================
    # Model Loading and Prediction
    # =============================================================================
    print(f"🎯 Loading model for prediction: {model_uri}")

    try:
        # Load model from MLflow registry
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded successfully")

        # Make predictions
        print("🔮 Executing batch predictions...")
        predictions = model.predict(X)
        print(f"✅ Generated {len(predictions)} predictions")

        # Add predictions back to the DataFrame
        data_pdf["prediction"] = predictions

        # Convert back to Spark DataFrame
        prediction_df = spark_session.createDataFrame(data_pdf)
        print(f"✅ Prediction completed successfully")

    except Exception as e:
        error_msg = f"Model prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    # =============================================================================
    # Output Data Preparation
    # =============================================================================
    print("📝 Preparing output with prediction metadata...")

    # Standardize output format with comprehensive metadata
    output_df = (
        prediction_df.withColumn(
            "prediction", prediction_df["prediction"].cast("string")
        )  # Cast prediction to string
        .withColumn("model_id", lit(model_version))  # Track model version
        .withColumn("timestamp", to_timestamp(lit(ts)))  # Prediction timestamp
        .withColumn("granularity", lit(granularity))  # Add granularity level
    )

    print("🎉 Batch prediction pipeline completed successfully")
    return output_df
