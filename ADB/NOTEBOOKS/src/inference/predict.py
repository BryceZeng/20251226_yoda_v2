import math
from datetime import timedelta, timezone

import mlflow
import pyspark.sql.functions as F
from pyspark.sql.functions import lit, struct, to_timestamp
from pyspark.sql.types import IntegerType


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    This is the same preprocessing logic used during model training.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def preprocess_raw_data(raw_df):
    """
    Preprocess raw taxi data to create the rounded timestamp columns required for feature store lookups.
    This applies the same transformation used during training.

    Args:
        raw_df: PySpark DataFrame with columns:
            - tpep_pickup_datetime (required for feature lookup)
            - tpep_dropoff_datetime (required for feature lookup)
            - Other columns like trip_distance, pickup_zip, dropoff_zip

    Returns:
        DataFrame with:
            - rounded_pickup_datetime (15-minute intervals)
            - rounded_dropoff_datetime (30-minute intervals)
            - Original columns (except original datetime columns)
    """
    # Check if preprocessing is needed
    if "rounded_pickup_datetime" in raw_df.columns and "rounded_dropoff_datetime" in raw_df.columns:
        print("Data already preprocessed - skipping transformation")
        return raw_df

    # Check if raw datetime columns exist
    if "tpep_pickup_datetime" not in raw_df.columns or "tpep_dropoff_datetime" not in raw_df.columns:
        raise ValueError(
            "Input data must contain 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' columns "
            "for feature store lookups. Please ensure raw data has these timestamp columns."
        )

    print("Preprocessing raw data: creating rounded timestamp columns for feature lookups...")

    # Apply the same preprocessing used during training
    processed_df = (
        raw_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    raw_df["tpep_pickup_datetime"], F.lit(15)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    raw_df["tpep_dropoff_datetime"], F.lit(30)
                )
            ),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )

    print("Preprocessing complete: rounded timestamps created")
    return processed_df


def predict_batch(
    spark_session, model_uri, input_table_name, model_version, ts
):
    """
    Simplified batch prediction using direct Spark SQL for feature retrieval.
    No more complex FeatureLookup - just straightforward SQL joins!

    Args:
        spark_session: Active Spark session
        model_uri: URI of the registered model (e.g., "models:/model_name@alias")
        input_table_name: Name of input table containing base data
        model_version: Version of the model being used
        ts: Timestamp for prediction metadata
    """

    mlflow.set_registry_uri("databricks-uc")

    # =============================================================================
    # SIMPLIFIED APPROACH: Direct SQL Feature Retrieval
    # =============================================================================
    # Instead of complex FeatureLookup, use simple Spark SQL to get features
    # This is much more readable and easier to debug!

    print(f"Loading and enriching data from: {input_table_name}")

    # Direct SQL approach - much simpler than FeatureLookup!
    enriched_df = spark_session.sql(
        f"""
        SELECT
            base.*,
            -- Pickup features: Join with feature table directly
            pickup.mean_fare_window_1h_pickup_zip,
            pickup.count_trips_window_1h_pickup_zip,

            -- Dropoff features: Join with feature table directly
            dropoff.count_trips_window_30m_dropoff_zip,
            dropoff.dropoff_is_weekend

        FROM {input_table_name} base

        -- Feature joins removed - using only simple_features now
    """
    )

    print(f"✅ Features joined successfully using direct SQL")
    print(f"📊 Enriched dataset shape: {enriched_df.count()} rows")

    # Load model and predict directly on enriched dataframe
    print(f"🚀 Running batch inference with model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Convert to Pandas for model prediction (if needed)
    # Most MLflow models work well with Spark DataFrames directly
    prediction_df = mlflow.pyfunc.spark_udf(spark_session, model_uri)(enriched_df)

    # Add metadata columns
    output_df = (
        prediction_df.withColumn("fare_amount", prediction_df["prediction"])
        .withColumn("model_id", lit(model_version))
        .withColumn("timestamp", to_timestamp(lit(ts)))
        .drop("prediction")
    )

    output_df.display()
    return output_df
