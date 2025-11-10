import mlflow
from pyspark.sql.functions import struct, lit, to_timestamp
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from datetime import timedelta, timezone
import math


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
    spark_session, model_uri, input_table_name, output_table_name, model_version, ts
):
    """
    Apply the model at the specified URI for batch inference on the table with name input_table_name,
    writing results to the table with name output_table_name.
    
    This function automatically handles preprocessing of raw data, so it can accept either:
    1. Raw data with tpep_pickup_datetime and tpep_dropoff_datetime (will be preprocessed)
    2. Already preprocessed data with rounded_pickup_datetime and rounded_dropoff_datetime
    
    Args:
        spark_session: Active Spark session
        model_uri: URI of the registered model (e.g., "models:/model_name@alias")
        input_table_name: Name of input table containing features
        output_table_name: Name of output table to write predictions
        model_version: Version of the model being used
        ts: Timestamp for prediction metadata
    """
    
    mlflow.set_registry_uri("databricks-uc")
    
    # Load input table
    print(f"Loading input data from: {input_table_name}")
    table = spark_session.table(input_table_name)
    
    # Automatically preprocess raw data if needed
    # This ensures the data has the rounded timestamp columns required for feature store lookups
    preprocessed_table = preprocess_raw_data(table)

    display(preprocessed_table)
       
    # Initialize Feature Engineering Client
    from databricks.feature_engineering import FeatureEngineeringClient    
    fe_client = FeatureEngineeringClient()
    
    print(f"Running batch inference with model: {model_uri}")
    # Score batch - Feature Store will automatically join required features
    prediction_df = fe_client.score_batch(model_uri=model_uri, df=preprocessed_table)
    
    # Add metadata columns
    output_df = (
        prediction_df.withColumn("fare_amount", prediction_df["prediction"])
        .withColumn("model_id", lit(model_version))
        .withColumn("timestamp", to_timestamp(lit(ts)))
        .drop("prediction")
    )
    
    print(f"Predictions generated. Writing to: {output_table_name}")
    output_df.display()

    # Model predictions are written to the Delta table provided as input.
    # Delta is the default format in Databricks Runtime 8.0 and above.
    output_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(output_table_name)
    
    print(f"Successfully wrote predictions to {output_table_name}")