# =============================================================================
# Generic ML Helper Functions for Databricks
# =============================================================================
# This module provides reusable utility functions for ML projects on Databricks.
# Key features:
# - Configuration management through YAML files
# - Generic timestamp rounding utilities
# - MLflow model version management
# - Flexible data preprocessing functions
#
# Usage:
# 1. Ensure your project has a Workflows/{env}-commons/look-up.yml configuration
# 2. Import this module: import helper as pp
# 3. Use functions as needed in your training/inference notebooks
# =============================================================================

import math
import os
from datetime import timedelta, timezone

import mlflow.pyfunc
import pyspark.sql.functions as F
import yaml
from mlflow import MlflowClient
from pyspark.sql.types import IntegerType


def load_config(env="dev", config_filename="look-up.yml"):
    """
    Load configuration from YAML file in standardized project structure.

    This function supports both Databricks workspace deployment and local development.
    It follows a standardized project structure where configuration files are stored
    in Workflows/{environment}-commons/ directories.

    Args:
        env (str): Environment name (dev, test, uat, prod). Defaults to "dev"
        config_filename (str): Name of the configuration file. Defaults to "look-up.yml"

    Returns:
        dict: Dictionary containing configuration variables with their default values.
              Returns empty dict if config file cannot be loaded.

    Project Structure Expected:
        project_root/
        ├── ADB/NOTEBOOKS/src/training/  (current file location)
        └── Workflows/
            ├── dev-commons/look-up.yml
            └── prod-commons/look-up.yml

    Configuration File Format:
        variables:
          VARIABLE_NAME:
            description: "Description of the variable"
            default: "default_value"
    """
    try:
        # Primary method: Try Databricks workspace path (when deployed)
        try:
            # Check if dbutils is available (Databricks environment)
            if "dbutils" in globals():
                workspace_root = (
                    "/Workspace"
                    + dbutils.notebook.entry_point.getDbutils()
                    .notebook()
                    .getContext()
                    .notebookPath()
                    .get()
                    .split("/src")[0]
                )
                config_path = (
                    f"{workspace_root}/Workflows/{env}-commons/{config_filename}"
                )
            else:
                raise Exception("dbutils not available, using fallback path")
        except Exception as e:
            print(f"Workspace path method failed: {e}")
            # Fallback: Use relative path from current notebook location
            # Calculate relative path from training directory to project root
            current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            config_path = os.path.join(
                current_dir,
                "../../../../../Workflows",
                f"{env}-commons",
                config_filename,
            )
            config_path = os.path.normpath(config_path)

        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract default values from configuration variables
        variables = {k: v.get('default') for k, v in config.get('variables', {}).items()}
        print(f"Successfully loaded {len(variables)} configuration variables")
        return variables

    except Exception as e:
        print(f"Warning: Could not load configuration file: {e}")
        print("Falling back to environment variables or widget parameters")
        return {}


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Round datetime to specified minute intervals and convert to Unix timestamp.

    This function is commonly used in feature engineering to create time windows
    for aggregating features (e.g., trip counts, average fares in 15-minute windows).

    Args:
        dt (datetime): Input datetime object to be rounded
        num_minutes (int): Interval in minutes to round to (default: 15)
                          Common values: 15, 30, 60 for different feature windows

    Returns:
        int: Unix timestamp of the rounded datetime in UTC

    Example:
        # Round to 15-minute intervals for pickup features
        rounded_timestamp = rounded_unix_timestamp(pickup_time, 15)

        # Round to 30-minute intervals for dropoff features
        rounded_timestamp = rounded_unix_timestamp(dropoff_time, 30)
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


# Create PySpark UDF for use in DataFrame operations
rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def create_rounded_timestamp_columns(
    df, datetime_columns, intervals, suffix_mapping=None
):
    """
    Generic function to add rounded timestamp columns to a DataFrame.

    This function allows you to create multiple rounded timestamp columns with different
    intervals, making it easy to join with various feature tables that use different
    time windows.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame containing datetime columns
        datetime_columns (list): List of datetime column names to process
        intervals (dict): Mapping of column names to rounding intervals in minutes
                         Example: {"pickup_datetime": 15, "dropoff_datetime": 30}
        suffix_mapping (dict, optional): Custom suffixes for rounded columns
                                       Example: {"pickup_datetime": "_pickup", "dropoff_datetime": "_dropoff"}

    Returns:
        pyspark.sql.DataFrame: DataFrame with additional rounded timestamp columns

    Example:
        # Create rounded timestamps for feature joins
        intervals = {"tpep_pickup_datetime": 15, "tpep_dropoff_datetime": 30}
        df_with_rounded = create_rounded_timestamp_columns(taxi_data,
                                                         ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                                                         intervals)
    """
    result_df = df

    for col_name in datetime_columns:
        if col_name not in intervals:
            continue

        interval = intervals[col_name]

        # Generate column name for rounded timestamp
        if suffix_mapping and col_name in suffix_mapping:
            new_col_name = f"rounded{suffix_mapping[col_name]}_datetime"
        else:
            # Default naming convention
            new_col_name = f"rounded_{col_name.replace('tpep_', '').replace('_datetime', '')}_datetime"

        result_df = result_df.withColumn(
            new_col_name,
            F.to_timestamp(
                rounded_unix_timestamp_udf(result_df[col_name], F.lit(interval))
            ),
        )

    return result_df


def rounded_taxi_data(
    taxi_data_df,
    pickup_interval=15,
    dropoff_interval=30,
    pickup_col="tpep_pickup_datetime",
    dropoff_col="tpep_dropoff_datetime",
):
    """
    Process taxi data by adding rounded timestamp columns for feature joins.

    This function is specifically designed for taxi/ride-sharing datasets but can be
    adapted for any dataset with pickup and dropoff timestamps. The rounded timestamps
    enable joining with feature store tables that aggregate data over time windows.

    Args:
        taxi_data_df (pyspark.sql.DataFrame): Input taxi trip data
        pickup_interval (int): Minutes to round pickup times to (default: 15)
        dropoff_interval (int): Minutes to round dropoff times to (default: 30)
        pickup_col (str): Name of pickup datetime column (default: "tpep_pickup_datetime")
        dropoff_col (str): Name of dropoff datetime column (default: "tpep_dropoff_datetime")

    Returns:
        pyspark.sql.DataFrame: Processed DataFrame with rounded timestamp columns,
                              original timestamp columns removed, and temp view created

    Example:
        # Standard taxi data processing
        processed_data = rounded_taxi_data(raw_taxi_data)

        # Custom intervals for different feature engineering requirements
        processed_data = rounded_taxi_data(raw_taxi_data,
                                         pickup_interval=10,
                                         dropoff_interval=45)
    """
    # Add rounded timestamp columns for feature store joins
    result_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df[pickup_col], F.lit(pickup_interval)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df[dropoff_col], F.lit(dropoff_interval)
                )
            ),
        )
        .drop(pickup_col)  # Remove original columns to avoid confusion
        .drop(dropoff_col)
    )

    # Create temporary view for SQL queries (common pattern in Databricks notebooks)
    result_df.createOrReplaceTempView("taxi_data")
    return result_df


def get_latest_model_version(model_name, registry_uri="databricks-uc"):
    """
    Get the latest version number for a registered MLflow model.

    This function scans all versions of a model in the MLflow registry and returns
    the highest version number. Useful for automated model deployment and versioning.

    Args:
        model_name (str): Full name of the model in format "catalog.schema.model_name"
                         Example: "pru.pac_mlops.pac_mlops-model"
        registry_uri (str): MLflow registry URI (default: "databricks-uc" for Unity Catalog)

    Returns:
        int: Latest version number of the model (starts from 1)

    Example:
        # Get latest version for model promotion
        latest_version = get_latest_model_version("pru.pac_mlops.taxi-fare-model")
        model_uri = f"models:/{model_name}/{latest_version}"
    """
    latest_version = 1

    try:
        mlflow_client = MlflowClient(registry_uri=registry_uri)
        for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
            version_int = int(mv.version)
            if version_int > latest_version:
                latest_version = version_int
    except Exception as e:
        print(f"Warning: Could not retrieve model versions for {model_name}: {e}")
        print("Returning default version 1")

    return latest_version


def get_config_value(config_dict, key, widget_name=None, default=None):
    """
    Get configuration value with fallback hierarchy: widget -> config -> default.

    This helper function implements a standard pattern for getting configuration values
    in Databricks notebooks, checking multiple sources in order of preference.

    Args:
        config_dict (dict): Configuration dictionary from load_config()
        key (str): Configuration key name
        widget_name (str, optional): Databricks widget name (if different from key)
        default: Default value if not found elsewhere

    Returns:
        Configuration value from the first available source

    Example:
        config = load_config("dev")
        schema = get_config_value(config, "SCHEMA", "SCHEMA", "default_schema")
        model_name = get_config_value(config, "MODEL_NAME", default="default_model")
    """
    # First try to get from Databricks widget (highest priority)
    if widget_name:
        try:
            # Check if dbutils is available (Databricks environment)
            if "dbutils" in globals():
                widget_value = dbutils.widgets.get(widget_name)
                if widget_value and widget_value.strip():
                    return widget_value.strip()
        except Exception as e:
            print(f"Warning: Could not get widget value for {widget_name}: {e}")

    # Fall back to configuration file
    if key in config_dict and config_dict[key] is not None:
        return config_dict[key]

    # Final fallback to default value
    return default
