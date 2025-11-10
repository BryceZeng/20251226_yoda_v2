# Databricks notebook source
# DBTITLE 1,Load autoreload extension
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Importing Python Libraries and Dependencies
import os
import helper as pp

from databricks.feature_engineering import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient

import mlflow
from mlflow.tracking import MlflowClient

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# COMMAND ----------

# DBTITLE 1,Notebook environment configuration variables
env = "dev"
config = pp.load_config(os.getenv("ENVIRONMENT", env))
print(f"Config loaded: {config}")
schema = config.get("SCHEMA")
catalog = config.get("CATALOG")
input_table_path = config.get("TRAINING_DATA_PATH")
model_name = config.get("MODEL_NAME")
experiment_name = config.get("EXPERIMENT_NAME")
pickup_features_table = config.get("PICKUP_FEATURES_TABLE")
dropoff_features_table = config.get("DROP_FEATURES_TABLE")

# COMMAND ----------

# DBTITLE 1,Set MLflow experiment and registry URI
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1,Prepare taxi data for analysis
raw_data = spark.read.format("csv").option("header",True).option("inferSchema",True).load(input_table_path)
taxi_data = pp.rounded_taxi_data(raw_data)

# COMMAND ----------

# DBTITLE 1,Create Pickup and Dropoff FeatureLookups
pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"],
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key=["rounded_dropoff_datetime"],
    ),
]

# COMMAND ----------

# DBTITLE 1,Prepare Training Data for Model Training
# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# Since the rounded timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

fe = FeatureEngineeringClient()

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fe.create_training_set(
    df=taxi_data, # specify the df 
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups, 
    # both features need to be available; defined in GenerateAndWriteFeatures &/or feature-engineering-workflow-resource.yml
    label="fare_amount",
    exclude_columns=exclude_columns,
)


# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureStoreClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

# DBTITLE 1,Train LightGBM model
features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)



# COMMAND ----------

display(X_train)

# COMMAND ----------

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------

# DBTITLE 1,Log test metrics and model parameters
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log test metrics to MLflow
mlflow.log_metric("test_mse", mse)
mlflow.log_metric("test_rmse", rmse)
mlflow.log_metric("test_mae", mae)
mlflow.log_metric("test_r2", r2)

# Log model parameters
mlflow.log_param("num_leaves", param["num_leaves"])
mlflow.log_param("objective", param["objective"])
mlflow.log_param("num_rounds", num_rounds)

# COMMAND ----------

# DBTITLE 1,Log trained model with feature lookup in MLflow
# Log the trained model with MLflow and package it with feature lookup information.
fe.log_model(
    model=model, #specify model
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,
)

# COMMAND ----------

# DBTITLE 1,Set model alias to Challenger
client = MlflowClient(registry_uri="databricks-uc")
model_version = pp.get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

client.set_registered_model_alias(
    name=model_name,
    version=model_version,
    alias="Challenger",
)

# COMMAND ----------

# DBTITLE 1,Set model deployment information
# The returned model URI is needed by the model deployment notebook.
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
