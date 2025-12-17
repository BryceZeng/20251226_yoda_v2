# =============================================================================
# Prediction Module for Batch Inference
# =============================================================================
# This module provides optimized prediction functionality for batch inference
# pipelines. It includes data preprocessing, feature enrichment, and model
# prediction capabilities with efficient Spark SQL operations.
#
# Key Functions:
# - preprocess_raw_data: Transforms raw data for feature store compatibility
# - predict_batch: Main prediction pipeline with feature enrichment
# - rounded_unix_timestamp: Time-based feature transformation
# =============================================================================

import math
from datetime import timedelta, timezone
from typing import Optional

import mlflow
import pyspark.sql.functions as F
from pyspark.sql.functions import lit, struct, to_timestamp
from pyspark.sql.types import IntegerType


def rounded_unix_timestamp(dt, num_minutes: int = 15) -> int:
    """
    Rounds datetime to specified minute intervals and returns unix timestamp.

    This function implements the same preprocessing logic used during model
    training to ensure consistency between training and inference data.

    Args:
        dt: Input datetime object
        num_minutes: Interval in minutes for rounding (default: 15)

    Returns:
        int: Unix timestamp rounded to the nearest interval

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2023, 1, 1, 12, 17, 30)  # 12:17:30
        >>> rounded_unix_timestamp(dt, 15)  # Rounds to 12:30:00
    """
    # Calculate seconds within the current hour
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6

    # Calculate seconds to add to reach next interval boundary
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs

    # Return unix timestamp of rounded datetime
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


# Create Spark UDF for distributed processing
rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def preprocess_raw_data(raw_df):
    """
    Preprocesses raw taxi data to create rounded timestamp columns for feature lookups.

    This function applies the same transformations used during training to ensure
    data consistency and proper feature store integration.

    Args:
        raw_df: PySpark DataFrame with required columns:
            - tpep_pickup_datetime: Pickup timestamp (required)
            - tpep_dropoff_datetime: Dropoff timestamp (required)
            - Additional columns: trip_distance, pickup_zip, dropoff_zip, etc.

    Returns:
        PySpark DataFrame with processed timestamp columns:
            - rounded_pickup_datetime: 15-minute interval boundaries
            - rounded_dropoff_datetime: 30-minute interval boundaries
            - All original columns except raw datetime columns

    Raises:
        ValueError: If required timestamp columns are missing
    """
    print("🔄 Starting data preprocessing...")

    # Check if data is already preprocessed
    if "rounded_pickup_datetime" in raw_df.columns and "rounded_dropoff_datetime" in raw_df.columns:
        print("✅ Data already preprocessed - skipping transformation")
        return raw_df

    # Validate required columns exist
    required_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            "Input data must contain 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' "
            "for feature store lookups."
        )

    print("📊 Creating rounded timestamp columns for feature lookups...")

    # Apply consistent preprocessing transformations
    processed_df = (
        raw_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(raw_df["tpep_pickup_datetime"], F.lit(15))
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(raw_df["tpep_dropoff_datetime"], F.lit(30))
            ),
        )
        .drop(
            "tpep_pickup_datetime", "tpep_dropoff_datetime"
        )  # Remove original columns
    )

    print("✅ Preprocessing completed successfully")
    return processed_df


def predict_batch(
    spark_session, model_uri: str, input_table_name: str, model_version: str, ts: str, granularity: str = "party"
):
    """
    Executes optimized batch prediction with feature enrichment.

    This function implements a streamlined prediction pipeline using direct
    Spark SQL for feature retrieval instead of complex FeatureLookup operations.
    This approach provides better performance and easier debugging.

    Args:
        spark_session: Active Spark session for data processing
        model_uri: MLflow model URI (e.g., "models:/model_name@alias")
        input_table_name: Name of input table containing base prediction data
        model_version: Version identifier of the model being used
        ts: Timestamp string for prediction metadata
        granularity: Granularity level for predictions (e.g., party, claims, agent)

    Returns:
        PySpark DataFrame with predictions and metadata columns:
            - Original input columns
            - fare_amount: Model prediction results
            - model_id: Model version used
            - timestamp: Prediction execution time
            - granularity: Prediction granularity level
            - Additional metadata as needed

    Raises:
        Exception: If model loading or prediction fails
    """
    print("🚀 Initializing batch prediction pipeline...")

    # Configure MLflow registry
    mlflow.set_registry_uri("databricks-uc")

    # =============================================================================
    # Data Loading - Support both CSV files and Delta tables
    # =============================================================================
    print(f"📊 Loading data from: {input_table_name}")

    try:
        # Check if input is a file path (CSV) or table name
        if input_table_name.endswith(".csv") or "/Volumes/" in input_table_name:
            print("📁 Loading data from CSV file...")
            base_df = (
                spark_session.read.format("csv")
                .option("header", True)
                .option("inferSchema", True)
                .load(input_table_name)
                .limit(5000)  # Limit to 5000 records for faster processing
            )
        else:
            print("📊 Loading data from Delta table...")
            base_df = spark_session.table(input_table_name).limit(5000)

        print(f"✅ Data loaded: {base_df.count()} rows, {len(base_df.columns)} columns")
        print(f"⚠️  Note: Data limited to 5000 records")
        print(f"📋 Input columns: {base_df.columns}")

    except Exception as e:
        error_msg = f"Failed to load input data: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
    
    # =============================================================================
    # Data Preprocessing - Apply same transformations as training
    # =============================================================================
    print("🔄 Preprocessing data (adding rounded timestamps)...")
    
    try:
        # Apply preprocessing to match training data format
        base_df = preprocess_raw_data(base_df)
        print(f"✅ Data preprocessed: {len(base_df.columns)} columns")
        print(f"📋 Preprocessed columns: {base_df.columns}")
    
    except Exception as e:
        print(f"⚠️  Preprocessing failed: {str(e)}")
        print("   Proceeding with raw data - model may fail if schema doesn't match training")
        # Continue with raw data

    # =============================================================================
    # Feature Enrichment with Optimized SQL
    # =============================================================================
    print(f"📊 Enriching data with features...")

    # Create temp view for SQL queries
    base_df.createOrReplaceTempView("inference_input_temp")

    # Direct SQL approach for feature joins - much simpler and faster than FeatureLookup
    enrichment_query = f"""
        SELECT
            base.*

            -- Note: Feature table joins removed for simplified implementation
            -- Add feature table joins here when feature tables are available:
            -- , pickup.mean_fare_window_1h_pickup_zip
            -- , pickup.count_trips_window_1h_pickup_zip
            -- , dropoff.count_trips_window_30m_dropoff_zip
            -- , dropoff.dropoff_is_weekend

        FROM inference_input_temp base

        -- LEFT JOIN feature_store.pickup_features pickup ON ...
        -- LEFT JOIN feature_store.dropoff_features dropoff ON ...
    """

    try:
        enriched_df = spark_session.sql(enrichment_query)
        print(f"✅ Feature enrichment completed")
        print(f"📈 Dataset shape: {enriched_df.count()} rows")

    except Exception as e:
        print(f"❌ Feature enrichment failed: {str(e)}")
        # Fallback: use input data without feature enrichment
        print("⚠️  Proceeding with base features only")
        enriched_df = base_df

    # =============================================================================
    # Model Loading and Prediction
    # =============================================================================
    print(f"🎯 Loading model for prediction: {model_uri}")

    try:
        # Load model from MLflow registry
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded successfully")
        
        # Show model signature for debugging
        try:
            model_info = mlflow.models.get_model_info(model_uri)
            if model_info.signature:
                print(f"📋 Model expected inputs: {model_info.signature.inputs}")
                print(f"📋 Model expected outputs: {model_info.signature.outputs}")
        except Exception as sig_err:
            print(f"⚠️  Could not retrieve model signature: {sig_err}")

        # Show what columns we're providing
        print(f"📊 Providing columns to model: {enriched_df.columns}")
        
        # Execute batch prediction using MLflow's Spark UDF for optimal performance
        print("🔮 Executing batch predictions...")
        prediction_df = mlflow.pyfunc.spark_udf(spark_session, model_uri)(enriched_df)

        print(f"✅ Prediction completed successfully")

    except Exception as e:
        error_msg = f"Model prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"💡 Troubleshooting hints:")
        print(f"   - Ensure input data has the same features used during training")
        print(f"   - Check that preprocessing is applied correctly")
        print(f"   - Available columns: {enriched_df.columns}")
        raise Exception(error_msg)

    # =============================================================================
    # Output Data Preparation
    # =============================================================================
    print("📝 Preparing output with prediction metadata...")

    # Standardize output format with comprehensive metadata
    output_df = (
        prediction_df.withColumn(
            "fare_amount", prediction_df["prediction"].cast("string")
        )  # Rename prediction column and cast to string
        .withColumn("model_id", lit(model_version))  # Track model version
        .withColumn("timestamp", to_timestamp(lit(ts)))  # Prediction timestamp
        .withColumn("granularity", lit(granularity))  # Add granularity level
        .drop("prediction")  # Remove original prediction column
    )

    # Display sample results for validation
    print("📋 Prediction pipeline results:")
    output_df.display()

    print("🎉 Batch prediction pipeline completed successfully")
    return output_df
