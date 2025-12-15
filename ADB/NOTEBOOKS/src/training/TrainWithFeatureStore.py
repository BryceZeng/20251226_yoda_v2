# Databricks notebook source
# =============================================================================
# Generic Model Training with Feature Store Integration
# =============================================================================
# This notebook provides a reusable template for training ML models with
# Databricks Feature Store integration. Key features:
#
# - Configuration-driven approach using YAML lookup files
# - Generic feature lookup creation based on configuration
# - MLflow experiment tracking and model registry integration
# - Flexible model training with configurable parameters
# - Automated model versioning and alias management
#
# To adapt for your project:
# 1. Update Workflows/{env}-commons/look-up.yml with your configuration
# 2. Modify feature lookup creation section for your feature tables
# 3. Adjust model training parameters as needed
# 4. Update model evaluation metrics if required
# =============================================================================

# DBTITLE 1,Load autoreload extension
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Import Required Libraries and Dependencies
# =============================================================================
# Standard Libraries
# =============================================================================
import os

# =============================================================================
# Databricks Utilities (ensure availability)
# =============================================================================
try:
    # dbutils should be available by default in Databricks notebooks
    dbutils
except NameError:
    print(
        "Warning: dbutils not available. Some functionality may not work in non-Databricks environments."
    )

# =============================================================================
# Project-Specific Helper Functions
# =============================================================================
import helper as pp

# =============================================================================
# Machine Learning Libraries
# =============================================================================
import lightgbm as lgb

# =============================================================================
# MLflow for Experiment Tracking and Model Registry
# =============================================================================
import mlflow
import mlflow.lightgbm
import numpy as np

# =============================================================================
# Databricks Feature Store
# =============================================================================
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# DBTITLE 1,Load Configuration and Environment Variables
# =============================================================================
# Configuration Loading Strategy:
# 1. Load base configuration from YAML file
# 2. Override with Databricks widget values (if provided)
# 3. Use intelligent defaults for missing values
#
# To customize for your project:
# - Update the configuration keys in Workflows/{env}-commons/look-up.yml
# - Add or remove configuration variables as needed
# - Modify default values to match your project requirements
# =============================================================================

# Load environment and configuration
default_env = "dev"
env = pp.get_config_value({}, "ENV", "ENV", default_env)
config = pp.load_config(env)
print(f"Environment: {env}")
print(f"Configuration loaded with {len(config)} variables")

# =============================================================================
# Core Configuration Variables
# =============================================================================
schema = pp.get_config_value(config, "SCHEMA", "SCHEMA", "pac_mlops")
catalog = pp.get_config_value(config, "CATALOG", "CATALOG", "pru")
input_table_path = pp.get_config_value(
    config, "TRAINING_DATA_PATH", "TRAINING_DATA_PATH", "/Volumes/pru/pac_mlops/test"
)
model_name = pp.get_config_value(
    config, "MODEL_NAME", "MODEL_NAME", "pru.pac_mlops.pac_mlops-model"
)
experiment_name = pp.get_config_value(
    config, "EXPERIMENT_NAME", "EXPERIMENT_NAME", "/dev-pac_mlops-experiment"
)

# Validate required configuration variables (with fallbacks)
if not experiment_name:
    print(
        "Warning: EXPERIMENT_NAME not found, using default: /dev-pac_mlops-experiment"
    )
    experiment_name = "/dev-pac_mlops-experiment"
if not model_name:
    print("Warning: MODEL_NAME not found, using default: pru.pac_mlops.pac_mlops-model")
    model_name = "pru.pac_mlops.pac_mlops-model"
if not input_table_path:
    print(
        "Warning: TRAINING_DATA_PATH not found, using default: /Volumes/pru/pac_mlops/test"
    )
    input_table_path = "/Volumes/pru/pac_mlops/test"

# =============================================================================
# Feature Store Configuration
# =============================================================================
# These variables define which feature tables to use for model training
# Modify these based on your feature engineering setup
# pickup_features_table and dropoff_features_table removed - using only simple_features now
# pickup_features_table = pp.get_config_value(
#     config,
#     "PICKUP_FEATURES_TABLE",
#     "PICKUP_FEATURES_TABLE",
#     "pru.pac_mlops.trip_pickup_features",
# )
# dropoff_features_table = pp.get_config_value(
#     config,
#     "DROP_FEATURES_TABLE",
#     "DROP_FEATURES_TABLE",
#     "pru.pac_mlops.trip_dropoff_features",
# )

# Print configuration for verification
print(f"\nConfiguration Summary:")
print(f"- Schema: {schema}")
print(f"- Catalog: {catalog}")
print(f"- Training Data Path: {input_table_path}")
print(f"- Model Name: {model_name}")
print(f"- Experiment Name: {experiment_name}")
# print(f"- Pickup Features Table: {pickup_features_table}")
# print(f"- Dropoff Features Table: {dropoff_features_table}")

# COMMAND ----------

# DBTITLE 1,Environment Safety Check
# =============================================================================
# Production Environment Protection
# =============================================================================
# This safety check prevents accidental model training in production environment.
# In production, model training should be handled through automated workflows
# with proper approval processes.
#
# To customize: Modify the condition based on your environment naming convention
# =============================================================================
if env == "prod":
    print("Production environment detected. Exiting to prevent accidental training.")
    print("Model training in production should be handled through approved workflows.")
    dbutils.notebook.exit(0)

# COMMAND ----------

# DBTITLE 1,Configure MLflow Experiment and Registry
# =============================================================================
# MLflow Configuration
# =============================================================================
# Set up MLflow for experiment tracking and model registry.
# This uses Unity Catalog for centralized model management.
#
# To customize:
# - Change registry URI if not using Unity Catalog
# - Modify experiment naming convention if needed
# =============================================================================
try:
    print(f"Setting up MLflow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    mlflow.set_registry_uri("databricks-uc")  # Use Unity Catalog for model registry

    print(f"MLflow experiment set to: {experiment_name}")
    print(f"Using registry URI: databricks-uc")
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    print(f"Experiment name provided: {experiment_name}")
    print(f"Type of experiment name: {type(experiment_name)}")
    raise

# COMMAND ----------

# DBTITLE 1,Load and Prepare Training Data
# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# Load raw training data and apply necessary transformations.
# The data preparation includes timestamp rounding for feature store joins.
#
# To customize for your project:
# - Modify data loading format if not using CSV
# - Update the data preparation function call
# - Add additional preprocessing steps as needed
# =============================================================================
print(f"Loading training data from: {input_table_path}")

# Load raw data (modify format as needed for your data source)
raw_data = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(input_table_path)
)

print(f"Raw data shape: {raw_data.count()} rows, {len(raw_data.columns)} columns")

# Apply data preprocessing (customize intervals based on your feature engineering)
taxi_data = pp.rounded_taxi_data(raw_data)

print("Data preprocessing completed - rounded timestamps added for feature joins")
print(f"Processed data columns: {taxi_data.columns}")

# COMMAND ----------

# DBTITLE 1,Create Feature Lookups for Model Training
# =============================================================================
# Feature Store Integration Setup
# =============================================================================
# Configure feature lookups to join training data with feature store tables.
# Each FeatureLookup defines:
# - table_name: Feature table in the feature store
# - feature_names: Specific features to retrieve
# - lookup_key: Keys for joining with training data
# - timestamp_lookup_key: For time-aware feature lookups
#
# To customize for your project:
# 1. Update table names to match your feature store tables
# 2. Modify feature names based on your feature engineering
# 3. Adjust lookup keys to match your data schema
# 4. Update timestamp keys for time-based features
# =============================================================================

# Pickup location features removed - using only simple_features now
# pickup_feature_lookups = [
#     FeatureLookup(
#         table_name=pickup_features_table,
#         feature_names=[
#             "mean_fare_window_1h_pickup_zip",  # Average fare in 1-hour window
#             "count_trips_window_1h_pickup_zip",  # Trip count in 1-hour window
#             # Add more pickup features here as needed:
#             # "median_trip_distance_pickup_zip",
#             # "std_fare_pickup_zip",
#         ],
#         lookup_key=["pickup_zip"],  # Join on pickup location
#         timestamp_lookup_key=["rounded_pickup_datetime"],  # Time-aware lookup
#     ),
# ]

# Dropoff location features removed - using only simple_features now
# dropoff_feature_lookups = [
#     FeatureLookup(
#         table_name=dropoff_features_table,
#         feature_names=[
#             "count_trips_window_30m_dropoff_zip",  # Trip count in 30-minute window
#             "dropoff_is_weekend",  # Weekend indicator
#             # Add more dropoff features here as needed:
#             # "popular_destination_score",
#             # "avg_wait_time_dropoff_zip",
#         ],
#         lookup_key=["dropoff_zip"],  # Join on dropoff location
#         timestamp_lookup_key=["rounded_dropoff_datetime"],  # Time-aware lookup
#     ),
# ]

# Feature lookups simplified - using only simple features
feature_lookups = []  # Empty for now, simple_features will be used directly

print("Feature lookups removed - using simple features approach")

# COMMAND ----------

# DBTITLE 1,Create Training Dataset with Feature Store Integration
# =============================================================================
# Training Set Creation
# =============================================================================
# Create a training dataset by joining raw data with feature store tables.
# The FeatureEngineeringClient handles the complex joins and ensures
# feature lineage tracking for model deployment.
#
# To customize for your project:
# - Update the label column name
# - Modify exclude_columns based on your data schema
# - Add additional feature lookups as needed
# =============================================================================

# End any existing MLflow runs to ensure clean state
mlflow.end_run()

# Start new MLflow run for experiment tracking
mlflow.start_run()

# Define columns to exclude from training to prevent data leakage
# These are typically timestamp columns that could cause overfitting
exclude_columns = [
    "rounded_pickup_datetime",
    "rounded_dropoff_datetime",
    # Add other columns to exclude here:
    # "trip_id",
    # "driver_id",
]

print(f"Excluding columns from training: {exclude_columns}")

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Create training set with simplified approach (no complex feature lookups)
print("Creating training set with simplified feature approach...")
training_set = fe.create_training_set(
    df=taxi_data,  # Base training data
    feature_lookups=feature_lookups,  # Empty feature lookups - using simple features instead
    label="fare_amount",  # Target variable (customize for your use case)
    exclude_columns=exclude_columns,
)

print("Training set created successfully")

# Load training set into DataFrame for model training
print("Loading training set into DataFrame...")
training_df = training_set.load_df()

print(
    f"Training dataset shape: {training_df.count()} rows, {len(training_df.columns)} columns"
)
print(f"Training dataset columns: {training_df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training and Evaluation
# MAGIC
# MAGIC This section covers:
# MAGIC - Data preparation for training
# MAGIC - Model training with configurable parameters
# MAGIC - Model evaluation and metrics logging
# MAGIC - Model registration and versioning
# MAGIC
# MAGIC The training process uses LightGBM by default but can be easily adapted for other algorithms.
# MAGIC To customize:
# MAGIC - Update model parameters in the training configuration
# MAGIC - Modify evaluation metrics based on your problem type
# MAGIC - Adjust train/test split ratio if needed

# COMMAND ----------

# DBTITLE 1,Prepare Data for Model Training
# =============================================================================
# Data Splitting and Preparation
# =============================================================================
# Convert Spark DataFrame to Pandas for scikit-learn compatibility
# and split into training and testing sets.
#
# To customize:
# - Modify test_size for different train/test ratios
# - Update random_state for reproducible results
# - Change target column name if different
# =============================================================================

features_and_label = training_df.columns
print(f"Total features and label columns: {len(features_and_label)}")
print(f"Columns: {features_and_label}")

# Convert to Pandas for scikit-learn compatibility
data = training_df.toPandas()[features_and_label]
print(f"Data converted to Pandas DataFrame: {data.shape}")

# Split data into training and testing sets
# Customize these parameters based on your requirements:
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 123  # For reproducible results
TARGET_COLUMN = "fare_amount"  # Update for your target variable

train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Separate features and target
X_train = train.drop([TARGET_COLUMN], axis=1)
X_test = test.drop([TARGET_COLUMN], axis=1)
y_train = train[TARGET_COLUMN]
y_test = test[TARGET_COLUMN]

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Enable MLflow autologging for automatic parameter and metric tracking
mlflow.lightgbm.autolog()

# Prepare LightGBM datasets
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

print("LightGBM datasets prepared successfully")


# COMMAND ----------

# DBTITLE 1,Preview Training Data
# =============================================================================
# Data Inspection
# =============================================================================
# Display sample of training features to verify data quality and feature values
# =============================================================================
print("Training data sample:")
print(f"Feature columns: {list(X_train.columns)}")
display(X_train.head(10))  # Show first 10 rows for inspection

# DBTITLE 1,Configure and Train LightGBM Model
# =============================================================================
# Model Training Configuration
# =============================================================================
# Define model parameters and training configuration.
# These parameters can be customized based on your specific use case.
#
# To customize:
# - Modify hyperparameters based on your data and problem type
# - Add additional parameters for more complex tuning
# - Consider using hyperparameter optimization frameworks
# =============================================================================

# Model hyperparameters (customize based on your requirements)
model_params = {
    "num_leaves": 32,  # Controls model complexity
    "objective": "regression",  # Change to 'classification' for classification tasks
    "metric": "rmse",  # Evaluation metric
    "boosting_type": "gbdt",  # Gradient boosting decision tree
    "feature_fraction": 0.9,  # Feature sampling ratio
    "bagging_fraction": 0.8,  # Data sampling ratio
    "bagging_freq": 5,  # Frequency of bagging
    "learning_rate": 0.1,  # Learning rate
    "verbose": -1,  # Suppress warnings
}

# Training configuration
NUM_ROUNDS = 100  # Number of boosting iterations
EARLY_STOPPING_ROUNDS = 10  # Stop if no improvement for N rounds

print("Model Configuration:")
for param, value in model_params.items():
    print(f"  {param}: {value}")
print(f"  num_rounds: {NUM_ROUNDS}")
print(f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")

# Train the model
print("\nStarting model training...")
model = lgb.train(
    model_params,
    train_lgb_dataset,
    num_boost_round=NUM_ROUNDS,
    valid_sets=[test_lgb_dataset],
    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS)],
)

print("Model training completed successfully")
print(f"Best iteration: {model.best_iteration}")

# COMMAND ----------

# DBTITLE 1,Evaluate Model and Log Metrics
# =============================================================================
# Model Evaluation and Metrics Logging
# =============================================================================
# Calculate evaluation metrics and log them to MLflow for experiment tracking.
# This includes both regression metrics and model parameters.
#
# To customize:
# - Add additional metrics specific to your problem type
# - Modify metrics for classification problems
# - Add custom evaluation logic if needed
# =============================================================================

# Make predictions on test set
print("Making predictions on test set...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Calculate evaluation metrics
print("Calculating evaluation metrics...")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate additional metrics
mean_y_test = np.mean(y_test)
mean_absolute_percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Print metrics for immediate feedback
print("\nModel Performance Metrics:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  R-squared (R²): {r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error:.2f}%")
print(f"  Test set mean: {mean_y_test:.4f}")

# Log metrics to MLflow
print("\nLogging metrics to MLflow...")
mlflow.log_metric("test_mse", mse)
mlflow.log_metric("test_rmse", rmse)
mlflow.log_metric("test_mae", mae)
mlflow.log_metric("test_r2", r2)
mlflow.log_metric("test_mape", mean_absolute_percentage_error)
mlflow.log_metric("test_mean", mean_y_test)

# Log model parameters
print("Logging model parameters to MLflow...")
for param, value in model_params.items():
    mlflow.log_param(param, value)
mlflow.log_param("num_rounds", NUM_ROUNDS)
mlflow.log_param("best_iteration", model.best_iteration)
mlflow.log_param("train_samples", X_train.shape[0])
mlflow.log_param("test_samples", X_test.shape[0])
mlflow.log_param("num_features", X_train.shape[1])

print("Metrics and parameters logged successfully")

# COMMAND ----------

# DBTITLE 1,Register Model with Feature Store Integration
# =============================================================================
# Model Registration and Feature Store Integration
# =============================================================================
# Log the trained model to MLflow with feature store metadata.
# This ensures that the model includes all necessary feature lookup information
# for deployment and inference.
#
# To customize:
# - Modify artifact_path if you want a different model artifact name
# - Update registered_model_name if using different naming conventions
# - Add model signature or input example if needed
# =============================================================================

print("Registering model with MLflow and Feature Store...")

# Log model with feature store integration
fe.log_model(
    model=model,  # Trained LightGBM model
    artifact_path="model_packaged",  # Artifact path in MLflow
    flavor=mlflow.lightgbm,  # Model flavor for LightGBM
    training_set=training_set,  # Feature store training set metadata
    registered_model_name=model_name,  # Model name in registry
)

print(f"Model successfully registered as: {model_name}")
print(
    "Model includes feature store metadata for automated feature lookups during inference"
)

# COMMAND ----------

# DBTITLE 1,Set Model Alias for Deployment Pipeline
# =============================================================================
# Model Alias Management
# =============================================================================
# Set model alias to "Challenger" for A/B testing and deployment workflows.
# This allows the deployment pipeline to identify new models for validation.
#
# Model Alias Strategy:
# - "Challenger": Newly trained model awaiting validation
# - "Champion": Current production model
# - "Staging": Model in staging environment
#
# To customize:
# - Modify alias name based on your deployment strategy
# - Add conditional logic for different environments
# - Implement automated alias promotion logic
# =============================================================================

# Get the latest model version
latest_version = pp.get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_version}"

print(f"Latest model version: {latest_version}")
print(f"Model URI: {model_uri}")

# Initialize MLflow client for model registry operations
client = MlflowClient(registry_uri="databricks-uc")

# Set model alias for deployment pipeline
alias_name = "Challenger"  # Customize based on your deployment strategy
print(f"Setting model alias '{alias_name}' for version {latest_version}")

client.set_registered_model_alias(
    name=model_name,
    version=latest_version,
    alias=alias_name,
)

print(f"Model alias '{alias_name}' set successfully")
print(f"Model is ready for validation and deployment workflows")

# COMMAND ----------

# DBTITLE 1,Export Model Information for Deployment Pipeline
# =============================================================================
# Deployment Information Export
# =============================================================================
# Set task values for downstream notebooks and workflows.
# These values are used by deployment and validation pipelines.
#
# Exported Information:
# - model_uri: Complete URI for model access
# - model_name: Registered model name
# - model_version: Specific model version number
#
# This information enables automated deployment workflows and model validation.
# =============================================================================

# Export model information for downstream tasks
print("Exporting model information for deployment pipeline...")

dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", latest_version)

print("\nDeployment Information:")
print(f"  Model URI: {model_uri}")
print(f"  Model Name: {model_name}")
print(f"  Model Version: {latest_version}")
print(f"  Model Alias: {alias_name}")

print("\nModel training and registration completed successfully!")
print("Model is ready for validation and deployment workflows.")

# Exit notebook with model URI for workflow integration
dbutils.notebook.exit(model_uri)
