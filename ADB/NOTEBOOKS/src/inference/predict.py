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
from datetime import timedelta, timezone
from typing import Optional

import mlflow
import pandas as pd
import pyspark.sql.functions as F
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
    catalog: str,
    schema: str,
    start_date: str,
    end_date: str,
    model_version: str,
    ts: str,
    granularity: str = "party",
):
    """
    Executes batch prediction using the same data loading and feature engineering as training.

    This function loads data using the SQL query from TrainWithFeatureStore.py and applies
    the same feature engineering transformations to ensure consistency.

    Args:
        spark_session: Active Spark session for data processing
        model_uri: MLflow model URI (e.g., "models:/model_name@alias")
        catalog: Databricks catalog name
        schema: Databricks schema name
        start_date: Start date for data filtering (YYYY-MM-DD)
        end_date: End date for data filtering (YYYY-MM-DD)
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
    # Data Loading - Use same SQL query as training
    # =============================================================================
    print(f"📊 Loading data from {catalog}.{schema}...")

    query = f"""
    WITH
    daterange AS (
      SELECT
        MIN(snapshot_date) as snapshot_date,
        QUARTER(snapshot_date) as trans_quarter,
        YEAR(snapshot_date) as trans_year
      FROM {catalog}.{schema}.customer_prumdm_daily_table
      WHERE snapshot_date BETWEEN '{start_date}' AND '{end_date}'
      GROUP BY QUARTER(snapshot_date), YEAR(snapshot_date)
      ORDER BY snapshot_date
    ),
    policy_filter AS (
      SELECT DISTINCT party_id
      FROM sdm.prumdm_enc.policy
      WHERE party_id LIKE 'LA%'
        AND (status = 'In Force' OR status = 'Paid Up Contract')
    ),
    table_a AS (
      SELECT
        a.*,
        DATE_FORMAT(ADD_MONTHS(a.snapshot_date, 3), 'yyyy-MM') as a_join_month
      FROM {catalog}.{schema}.customer_prumdm_daily_table a
      INNER JOIN daterange d ON a.snapshot_date = d.snapshot_date
      INNER JOIN policy_filter p ON a.party_id = p.party_id
      WHERE a.party_id LIKE 'LA%'
    ),
    table_b AS (
      SELECT
        b.party_id,
        b.ci_purchase_ind,
        b.medical_purchase_ind,
        b.protection_purchase_ind,
        b.savings_purchase_ind,
        b.investment_purchase_ind,
        b.retirement_purchase_ind,
        b.legacy_planning_purchase_ind,
        DATE_FORMAT(b.snapshot_date, 'yyyy-MM') as b_snapshot_month
      FROM {catalog}.{schema}.customer_target_yoda_daily_table b
      INNER JOIN policy_filter p ON b.party_id = p.party_id
      WHERE b.party_id LIKE 'LA%'
    )
    SELECT
      a.party_id,
      a.snapshot_date,
      DATE_FORMAT(a.snapshot_date, 'yyyy-MM') as trans_yyyymm,
      b.ci_purchase_ind,
      b.medical_purchase_ind,
      b.protection_purchase_ind,
      b.savings_purchase_ind,
      b.investment_purchase_ind,
      b.retirement_purchase_ind,
      b.legacy_planning_purchase_ind,
      a.*
    FROM table_a a
    INNER JOIN table_b b
      ON a.party_id = b.party_id
      AND a.a_join_month = b.b_snapshot_month
    """

    try:
        base_df = spark_session.sql(query)
        print(f"✅ Data loaded: {base_df.count()} rows, {len(base_df.columns)} columns")
    except Exception as e:
        error_msg = f"Failed to load data: {str(e)}"
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
