"""
================================================================================
Feature Engineering Pipeline - Generate and Write Features
================================================================================

This notebook generates features from raw data and writes them to a Databricks
Feature Store table. It's designed to be executed as part of an automated MLOps
pipeline defined in the feature-engineering workflow configuration.

Key Capabilities:
- Dynamic feature computation using pluggable transform modules
- Date-based filtering for incremental processing
- Automatic feature table creation with proper schema
- Merge-based writes to handle updates and new records

Workflow Integration:
Configured for execution in: pac_mlops/resources/feature-engineering-workflow-resource.yml

Required Parameters:
- input_table_path: Path to source data (Delta table, CSV, etc.)
- output_table_name: Fully qualified feature table name (catalog.schema.table)
- primary_keys: Comma-separated list of primary key columns
- features_transform_module: Python module with feature transform logic

Optional Parameters:
- timestamp_column: Column used for temporal filtering and as timestamp key
- input_start_date: Start date for data filtering (YYYY-MM-DD format)
- input_end_date: End date for data filtering (YYYY-MM-DD format)

Author: MLOps Team
Last Modified: December 2025
================================================================================
"""

# =============================================================================
# WIDGET CONFIGURATION - Notebook Parameter Definitions
# =============================================================================

# Default values for development and testing
DEFAULT_INPUT_PATH = "/databricks-datasets/nyctaxi-with-zipcodes/subsampled"
DEFAULT_OUTPUT_TABLE = "pru.mlops.trip_simple_features"
DEFAULT_TIMESTAMP_COLUMN = "rounded_datetime"
DEFAULT_TRANSFORM_MODULE = "simple_features"
DEFAULT_PRIMARY_KEYS = "zip,dropoff_zip"

# Required Parameters
dbutils.widgets.text(
    "input_table_path",
    DEFAULT_INPUT_PATH,
    label="Input Table Path (Required)"
)

dbutils.widgets.text(
    "output_table_name",
    DEFAULT_OUTPUT_TABLE,
    label="Output Feature Table Name (Required)"
)

dbutils.widgets.text(
    "primary_keys",
    DEFAULT_PRIMARY_KEYS,
    label="Primary Key Columns (Required, comma-separated)"
)

dbutils.widgets.text(
    "features_transform_module",
    DEFAULT_TRANSFORM_MODULE,
    label="Feature Transform Module Name (Required)"
)

# Optional Parameters for Date Filtering
dbutils.widgets.text(
    "input_start_date",
    "",
    label="Input Start Date (Optional, YYYY-MM-DD)"
)

dbutils.widgets.text(
    "input_end_date",
    "",
    label="Input End Date (Optional, YYYY-MM-DD)"
)

dbutils.widgets.text(
    "timestamp_column",
    DEFAULT_TIMESTAMP_COLUMN,
    label="Timestamp Column (Optional, for date filtering)"
)

# COMMAND ----------

# =============================================================================
# ENVIRONMENT SETUP - Configure Notebook Path and Dependencies
# =============================================================================

import os
import sys

# Get the current notebook path and navigate to the features module directory
# This ensures the features transform modules can be imported correctly
try:
    notebook_path = '/Workspace/' + os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    print(f"Current notebook path: {notebook_path}")

    # Change to the notebook directory first, then navigate to features
    %cd $notebook_path
    %cd ../features

    # Verify we're in the correct directory
    current_dir = %pwd
    print(f"Features module directory: {current_dir}")

except Exception as e:
    print(f"Warning: Could not set up notebook path: {e}")
    print("Proceeding with current directory...")

# COMMAND ----------

# =============================================================================
# PARAMETER VALIDATION AND SETUP
# =============================================================================

def validate_required_parameters():
    """
    Validate all required parameters are provided and have valid values.

    Returns:
        dict: Dictionary of validated parameters

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    params = {}

    # Required parameters validation
    required_params = {
        "input_table_path": "Input table path must be specified",
        "output_table_name": "Output table name must be specified",
        "primary_keys": "Primary keys must be specified",
        "features_transform_module": "Features transform module must be specified"
    }

    for param_name, error_msg in required_params.items():
        value = dbutils.widgets.get(param_name).strip()
        if not value:
            raise ValueError(f"❌ {error_msg}")
        params[param_name] = value

    # Optional parameters
    optional_params = ["input_start_date", "input_end_date", "timestamp_column"]
    for param_name in optional_params:
        params[param_name] = dbutils.widgets.get(param_name).strip()

    return params

# Validate and extract parameters
print("🔍 Validating notebook parameters...")
params = validate_required_parameters()

# Extract parameters into individual variables for backward compatibility
input_table_path = params["input_table_path"]
output_table_name = params["output_table_name"]
input_start_date = params["input_start_date"]
input_end_date = params["input_end_date"]
ts_column = params["timestamp_column"]
features_module = params["features_transform_module"]
pk_columns = params["primary_keys"]

# Validate output table name format (should be catalog.schema.table)
table_parts = output_table_name.split(".")
if len(table_parts) < 3:
    print("⚠️  Warning: Output table name format should be 'catalog.schema.table' for Unity Catalog")
    # Extract database/schema name for backward compatibility
    output_database = table_parts[1] if len(table_parts) >= 2 else table_parts[0]
else:
    output_database = table_parts[1]  # Schema name in Unity Catalog

# Log configuration summary
print("\n📋 Configuration Summary:")
print(f"   Input Table: {input_table_path}")
print(f"   Output Table: {output_table_name}")
print(f"   Database/Schema: {output_database}")
print(f"   Primary Keys: {pk_columns}")
print(f"   Transform Module: {features_module}")
print(f"   Timestamp Column: {ts_column or 'None'}")
print(f"   Date Range: {input_start_date or 'None'} to {input_end_date or 'None'}")

# COMMAND ----------

# =============================================================================
# DATABASE PREPARATION
# =============================================================================

print(f"💾 Creating database/schema '{output_database}' if it doesn't exist...")
try:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_database}")
    print(f"✅ Database/schema '{output_database}' is ready")
except Exception as e:
    print(f"❌ Error creating database '{output_database}': {e}")
    raise

# COMMAND ----------

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_input_data(table_path):
    """
    Load input data with automatic format detection and error handling.

    Args:
        table_path (str): Path to the input data

    Returns:
        DataFrame: Loaded input data
    """
    print(f"📁 Loading input data from: {table_path}")

    try:
        # Try to read as Delta table first (most common in Databricks)
        if table_path.startswith("dbfs:") or table_path.startswith("/databricks") or not "." in os.path.basename(table_path):
            try:
                raw_data = spark.read.format("delta").load(table_path)
                print(f"✅ Successfully loaded Delta table")
                return raw_data
            except Exception as delta_error:
                print(f"⚠️  Delta format not found, trying CSV format...")
                pass

        # If Delta fails, try CSV format
        if table_path.endswith('.csv') or '/subsampled' in table_path or '/test' in table_path:
            try:
                raw_data = (
                    spark.read
                    .format("csv")
                    .option("header", True)
                    .option("inferSchema", True)
                    .load(table_path)
                )
                print(f"✅ Successfully loaded CSV data")
                return raw_data
            except Exception as csv_error:
                print(f"⚠️  CSV format failed, trying table read...")
                pass

        # Default to table read
        raw_data = spark.table(table_path)
        print(f"✅ Successfully loaded as table")
        return raw_data

    except Exception as e:
        print(f"❌ Error loading data from {table_path}: {e}")
        raise

# Load the input data
raw_data = load_input_data(input_table_path)

# Display basic information about the loaded data
row_count = raw_data.count()
column_count = len(raw_data.columns)
print(f"📊 Data Summary: {row_count:,} rows, {column_count} columns")
print(f"   Columns: {', '.join(raw_data.columns[:10])}{'...' if column_count > 10 else ''}")

# COMMAND ----------

# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def load_transform_module(module_name):
    """
    Dynamically load the feature transformation module.

    Args:
        module_name (str): Name of the module to import

    Returns:
        function: The compute_features_fn function from the module
    """
    print(f"🔄 Loading feature transform module: {module_name}")

    try:
        from importlib import import_module
        mod = import_module(module_name)

        if not hasattr(mod, 'compute_features_fn'):
            raise AttributeError(f"Module '{module_name}' must contain a 'compute_features_fn' function")

        compute_features_fn = getattr(mod, "compute_features_fn")
        print(f"✅ Successfully loaded transform function from {module_name}")
        return compute_features_fn

    except ImportError as e:
        print(f"❌ Error importing module '{module_name}': {e}")
        print(f"   Make sure the module exists in the features directory")
        raise
    except Exception as e:
        print(f"❌ Error loading transform function: {e}")
        raise

# Load the feature transformation function
compute_features_fn = load_transform_module(features_module)

# Compute features with comprehensive logging
print(f"⚙️ Computing features using {features_module}.compute_features_fn...")
print(f"   Timestamp column: {ts_column or 'None'}")
print(f"   Date filter: {input_start_date or 'None'} to {input_end_date or 'None'}")

try:
    features_df = compute_features_fn(
        input_df=raw_data,
        timestamp_column=ts_column if ts_column else None,
        start_date=input_start_date if input_start_date else None,
        end_date=input_end_date if input_end_date else None,
    )

    # Validate the computed features
    feature_count = features_df.count()
    feature_columns = len(features_df.columns)

    print(f"✅ Feature computation completed successfully")
    print(f"   Generated {feature_count:,} feature records with {feature_columns} columns")
    print(f"   Feature columns: {', '.join(features_df.columns[:10])}{'...' if feature_columns > 10 else ''}")

    if feature_count == 0:
        print("⚠️  Warning: No feature records generated. Check your date filters and input data.")

except Exception as e:
    print(f"❌ Error during feature computation: {e}")
    raise

# COMMAND ----------

# =============================================================================
# FEATURE STORE OPERATIONS
# =============================================================================

def setup_feature_table(fe_client, table_name, primary_keys, timestamp_key, features_df):
    """
    Create feature table if it doesn't exist with proper configuration.

    Args:
        fe_client: FeatureEngineeringClient instance
        table_name (str): Fully qualified table name
        primary_keys (list): List of primary key columns
        timestamp_key (str): Timestamp column name
        features_df: DataFrame with feature data
    """
    print(f"🛠️ Setting up feature table: {table_name}")

    try:
        # Parse primary keys and include timestamp column if specified
        pk_list = [key.strip() for key in primary_keys.split(",") if key.strip()]

        if timestamp_key and timestamp_key not in pk_list:
            pk_list.append(timestamp_key)

        timestamp_keys = [timestamp_key] if timestamp_key else None

        print(f"   Primary keys: {pk_list}")
        print(f"   Timestamp keys: {timestamp_keys}")

        # Create the feature table (no-op if already exists with same schema)
        fe_client.create_table(
            name=table_name,
            primary_keys=pk_list,
            timestamp_keys=timestamp_keys,
            df=features_df,
        )

        print(f"✅ Feature table setup completed")

    except Exception as e:
        print(f"❌ Error setting up feature table: {e}")
        raise

def write_features_to_store(fe_client, table_name, features_df):
    """
    Write features to the feature store with merge mode.

    Args:
        fe_client: FeatureEngineeringClient instance
        table_name (str): Fully qualified table name
        features_df: DataFrame with feature data to write
    """
    print(f"📝 Writing features to feature store...")

    try:
        fe_client.write_table(
            name=table_name,
            df=features_df,
            mode="merge",
        )

        print(f"✅ Successfully wrote features to {table_name}")

    except Exception as e:
        print(f"❌ Error writing features to store: {e}")
        raise

# Initialize Feature Engineering Client
print("🔗 Initializing Databricks Feature Engineering client...")
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("✅ Feature Engineering client initialized")
except Exception as e:
    print(f"❌ Error initializing Feature Engineering client: {e}")
    raise

# Setup feature table and write features
setup_feature_table(fe, output_table_name, pk_columns, ts_column, features_df)
write_features_to_store(fe, output_table_name, features_df)

# COMMAND ----------

# =============================================================================
# PIPELINE COMPLETION AND SUMMARY
# =============================================================================

print("✨ Feature engineering pipeline completed successfully!")
print("\n📊 Final Summary:")
print(f"   • Input Source: {input_table_path}")
print(f"   • Output Table: {output_table_name}")
print(f"   • Features Generated: {features_df.count():,} records")
print(f"   • Transform Module: {features_module}")
print(f"   • Primary Keys: {pk_columns}")
if ts_column:
    print(f"   • Timestamp Column: {ts_column}")
if input_start_date or input_end_date:
    print(f"   • Date Range: {input_start_date or 'None'} to {input_end_date or 'None'}")

print("\n🎆 Pipeline execution completed. Features are now available in the Feature Store!")

# Exit with success status
dbutils.notebook.exit(0)
