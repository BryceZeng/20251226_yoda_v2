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
    prediction_col: str = "prediction",
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
        prediction_col: Name of the prediction column to create (default: "prediction")

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
                SELECT *, DATE_FORMAT(snapshot_date, 'yyyy-MM') as trans_yyyymm
                FROM {input_table_name}
                WHERE snapshot_date >= '{start_date_str}'
                  AND snapshot_date <= '{today}'
            """
            print(f"📆 Filtering data from {start_date_str} to {today}")
        else:
            # Get all available data if no existing predictions
            query = f"""
                SELECT *, DATE_FORMAT(snapshot_date, 'yyyy-MM') as trans_yyyymm
                FROM {input_table_name}
            """
            print(f"📆 Processing all available data (no existing predictions)")

        base_df = spark_session.sql(query)

        # Get row count for partition calculation
        row_count = base_df.count()
        col_count = len(base_df.columns)
        print(f"✅ Data loaded: {row_count} rows, {col_count} columns")

        if row_count == 0:
            print(f"⚠️  WARNING: No new data to process!")
            print(
                f"   Either input table '{input_table_name}' is empty or all data has been processed"
            )
            return spark_session.createDataFrame([], base_df.schema)

        print(f"✅ trans_yyyymm column created from snapshot_date in SQL query")

    except Exception as e:
        error_msg = f"Failed to load data from '{input_table_name}': {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    # =============================================================================
    # Feature Engineering - Apply same transformations as training
    # =============================================================================
    print("🔄 Applying feature engineering and predictions in batches...")

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

    # NOTE: Do NOT filter feature_cols here!
    # Some features are engineered during predict_batch_udf execution
    # (policy_count, age, months_since_consent)
    print(
        f"📊 Using {len(feature_cols)} features for prediction (includes engineered features)"
    )

    # Check what base columns we have
    print(f"📋 Base DataFrame columns: {base_df.columns}")

    # =============================================================================
    # Model Loading and Batch Prediction with pandas_udf
    # =============================================================================
    print(f"🎯 Loading model for prediction: {model_uri}")

    try:
        # Load model from MLflow registry on driver
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded successfully")

        # Broadcast the model to all executors for efficient distributed processing
        broadcasted_model = spark_session.sparkContext.broadcast(model)
        print(f"📡 Model broadcasted to executors")

        # Define the schema for the output
        import pandas as pd
        from pyspark.sql.functions import PandasUDFType, pandas_udf
        from pyspark.sql.types import StringType, StructField, TimestampType

        # Get the input schema and add prediction columns
        # Use the prediction_col parameter name instead of hardcoding "prediction"
        output_schema = base_df.schema.add(
            StructField(prediction_col, StringType(), True)
        )
        output_schema = output_schema.add(StructField("model_id", StringType(), True))
        output_schema = output_schema.add(
            StructField("timestamp", TimestampType(), True)
        )
        output_schema = output_schema.add(
            StructField("granularity", StringType(), True)
        )

        # Get the list of expected output columns (original columns + prediction columns)
        expected_output_cols = [field.name for field in output_schema.fields]

        # Function to apply feature engineering and predictions using pandas_udf
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def predict_batch_udf(batch_pdf):
            """
            Process each batch independently using the broadcasted model.
            This function runs on executors with the broadcasted model available.
            """
            if len(batch_pdf) == 0:
                return batch_pdf

            # Debug: Check what columns we have before feature engineering
            import sys
            print(f"DEBUG: Columns before engineering: {batch_pdf.columns.tolist()}", file=sys.stderr)
            print(f"DEBUG: Has trans_yyyymm: {'trans_yyyymm' in batch_pdf.columns}", file=sys.stderr)
            print(f"DEBUG: Has date_of_birth: {'date_of_birth' in batch_pdf.columns}", file=sys.stderr)
            print(f"DEBUG: Has policy_number: {'policy_number' in batch_pdf.columns}", file=sys.stderr)

            # Ensure trans_yyyymm is string format for pandas processing
            if "trans_yyyymm" in batch_pdf.columns:
                batch_pdf["trans_yyyymm"] = batch_pdf["trans_yyyymm"].astype(str)

            # Apply feature engineering to this batch
            batch_pdf = engineer_features(batch_pdf)

            # Debug: Check what columns we have after feature engineering
            print(f"DEBUG: Columns after engineering: {batch_pdf.columns.tolist()}", file=sys.stderr)
            print(f"DEBUG: Has policy_count: {'policy_count' in batch_pdf.columns}", file=sys.stderr)
            print(f"DEBUG: Has age: {'age' in batch_pdf.columns}", file=sys.stderr)
            print(f"DEBUG: Has months_since_consent: {'months_since_consent' in batch_pdf.columns}", file=sys.stderr)

            # Ensure all required feature columns are present
            # If any are missing after engineering, create them with default values
            missing_cols = []
            for col in feature_cols:
                if col not in batch_pdf.columns:
                    missing_cols.append(col)
                    # Determine appropriate default value based on column type
                    if col in [
                        "gender",
                        "marital_status",
                        "segment_description",
                        "microsegment",
                    ]:
                        batch_pdf[col] = "UNKNOWN"
                    elif col in [
                        "is_smoker",
                        "hazardous_lifestyle_ind",
                        "communication_consent",
                        "policy_count",
                    ]:
                        batch_pdf[col] = 0
                    else:  # numerical columns (age, months_since_consent, etc.)
                        batch_pdf[col] = -1.0

            if missing_cols:
                print(f"DEBUG: Created missing columns with defaults: {missing_cols}", file=sys.stderr)

            # Ensure correct data types for all features
            # Categorical features as strings
            for col in [
                "gender",
                "marital_status",
                "segment_description",
                "microsegment",
            ]:
                if col in batch_pdf.columns:
                    batch_pdf[col] = batch_pdf[col].astype(str)

            # Integer features
            for col in [
                "is_smoker",
                "hazardous_lifestyle_ind",
                "communication_consent",
                "policy_count",
            ]:
                if col in batch_pdf.columns:
                    batch_pdf[col] = batch_pdf[col].astype(int)

            # Float features
            for col in [
                "salary",
                "bmi",
                "ctp_value",
                "upgrader_shortfall",
                "age",
                "months_since_consent",
            ]:
                if col in batch_pdf.columns:
                    batch_pdf[col] = batch_pdf[col].astype(float)

            # Extract features for prediction - now all columns should exist with correct types
            X_batch = batch_pdf[feature_cols]
            print(f"DEBUG: X_batch shape: {X_batch.shape}, columns: {X_batch.columns.tolist()}", file=sys.stderr)

            # Get the broadcasted model and make predictions
            model_obj = broadcasted_model.value
            predictions = model_obj.predict(X_batch)

            # Add prediction columns using the parameter name
            batch_pdf[prediction_col] = predictions.astype(str)
            batch_pdf["model_id"] = model_version
            batch_pdf["timestamp"] = pd.to_datetime(ts)
            batch_pdf["granularity"] = granularity

            # Return only the columns that match the output schema
            # This drops engineered feature columns and temporary columns
            return batch_pdf[expected_output_cols]

        # Add a partition key for grouping (process in chunks based on row number)
        # This ensures we process data in manageable batches
        from pyspark.sql.functions import floor, monotonically_increasing_id

        batch_size = 10000  # Process 10k rows at a time

        df_with_batch_id = base_df.withColumn("_row_id", monotonically_increasing_id())
        df_with_batch_id = df_with_batch_id.withColumn(
            "_batch_id", floor(df_with_batch_id["_row_id"] / batch_size)
        )

        print(f"🔀 Processing data in batches of {batch_size} rows")

        # Apply the UDF to each batch group
        print("🔮 Executing distributed batch predictions across executors...")
        prediction_df = df_with_batch_id.groupby("_batch_id").apply(predict_batch_udf)

        # Drop the helper columns
        prediction_df = prediction_df.drop("_row_id", "_batch_id")

        print(f"✅ Distributed prediction pipeline configured successfully")

    except Exception as e:
        error_msg = f"Model prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

    # =============================================================================
    # Output Data Preparation
    # =============================================================================
    print("📝 Output ready with prediction metadata...")

    # The predictions are already prepared with all necessary columns
    # No need to convert or add more columns - everything is done in the partition processing
    output_df = prediction_df

    print("🎉 Batch prediction pipeline completed successfully")
    return output_df
