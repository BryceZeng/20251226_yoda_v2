# Databricks notebook source
# DBTITLE 1,Load autoreload extension
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions>=4.7.0 "mlflow[databricks]>=3.1" catboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions>=4.7.0

# COMMAND ----------

# DBTITLE 1,Importing Python Libraries and Dependencies
import os
from datetime import datetime

import helper as pp
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# DBTITLE 1,Notebook environment configuration variables
dbutils.widgets.text("SCHEMA", "prj_yoda", "Schema")
dbutils.widgets.text("ENV", "dev", "Environment")
dbutils.widgets.text("CATALOG", "ai_engineering", "Catalog")
dbutils.widgets.text("MODEL_NAME", "my_model", "Model Name")
dbutils.widgets.text("EXPERIMENT_NAME", "my_experiment", "Experiment Name")

schema = dbutils.widgets.get("SCHEMA")
env = dbutils.widgets.get("ENV")
catalog = dbutils.widgets.get("CATALOG")
model_name = dbutils.widgets.get("MODEL_NAME")
experiment_name = dbutils.widgets.get("EXPERIMENT_NAME")

# COMMAND ----------

if env == "prod":
    dbutils.notebook.exit(0)

# COMMAND ----------

# DBTITLE 1,Set MLflow experiment and registry URI
experiment_name = "/Shared/pacs-743-nba_yoda_v2"
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DBTITLE 1,Load data from SQL query
query = """
WITH
daterange AS (
  SELECT
    MIN(snapshot_date) as snapshot_date,
    QUARTER(snapshot_date) as trans_quarter,
    YEAR(snapshot_date) as trans_year
  FROM ai_engineering.prj_yoda.customer_prumdm_daily_table
  WHERE snapshot_date BETWEEN '2023-10-01' AND '2025-06-01'
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
  FROM ai_engineering.prj_yoda.customer_prumdm_daily_table a
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
  FROM ai_engineering.prj_yoda.customer_target_yoda_daily_table b
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

raw_data = spark.sql(query)
data_df = raw_data.toPandas()

# COMMAND ----------

# DBTITLE 1,Feature Engineering

def engineer_features(df):
    """Engineer features from raw data with null handling"""
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


data_df = engineer_features(data_df)

# COMMAND ----------

# DBTITLE 1,Prepare data for multiclass classification
# Define target columns
target_cols = [
    "ci_purchase_ind",
    "medical_purchase_ind",
    "protection_purchase_ind",
    "savings_purchase_ind",
    "investment_purchase_ind",
    "retirement_purchase_ind",
    "legacy_planning_purchase_ind",
]


# Create multiclass target
def create_multiclass_target(row):
    for idx, col in enumerate(target_cols, 1):
        if row[col] == 1:
            return idx
    return 0


data_df["target"] = data_df.apply(create_multiclass_target, axis=1)

# Define feature columns
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
feature_cols = [col for col in feature_cols if col in data_df.columns]

X = data_df[feature_cols]
y = data_df["target"]

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"\nNull values in features:\n{X.isnull().sum()}")

# COMMAND ----------

# DBTITLE 1,Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# COMMAND ----------

# DBTITLE 1,Train CatBoost multiclass model
mlflow.end_run()
mlflow.start_run()

# Identify categorical features
cat_features = ["gender", "marital_status", "segment_description", "microsegment"]
cat_features = [col for col in cat_features if col in X_train.columns]

print(f"Categorical features: {cat_features}")

# Create CatBoost pools
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Model parameters
params = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "random_seed": 123,
    "verbose": 100,
    "early_stopping_rounds": 50,
}

# Train model
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=test_pool)

# COMMAND ----------

# DBTITLE 1,Evaluate and log metrics
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_f1_macro", f1_macro)
mlflow.log_metric("test_f1_weighted", f1_weighted)
mlflow.log_params(params)

# Log feature importance
feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")
print("\nTop 10 Features:")
print(feature_importance.head(10))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# COMMAND ----------

# DBTITLE 1,Log model to MLflow
# Set the registry URI to Unity Catalog (recommended)
mlflow.set_registry_uri("databricks-uc")

# Construct the full 3-level Unity Catalog model name
full_model_name = f"{catalog}.{schema}.{model_name}"

# Debug: Print the constructed model name
print(f"🔍 Debug Information:")
print(f"   Catalog: {catalog}")
print(f"   Schema: {schema}")
print(f"   Model Name: {model_name}")
print(f"   Full Model Name: {full_model_name}")

from mlflow.models import infer_signature

# Example: X_train is your training data, model is your trained CatBoost model
signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train.iloc[[0]]  # or use a representative sample

print(f"📝 Logging model to MLflow with name: {full_model_name}")

mlflow.catboost.log_model(
    model,
    name="yoda_model",
    registered_model_name=full_model_name,
    signature=signature,
    input_example=input_example
)

print(f"✅ Model logged successfully!")

# COMMAND ----------

# DBTITLE 1,Set model alias to Challenger
# Use the full_model_name constructed above (don't redefine model_name)
print(f"🏷️  Setting Challenger alias for model: {full_model_name}")

client = MlflowClient(registry_uri="databricks-uc")
model_version = pp.get_latest_model_version(full_model_name)
model_uri = f"models:/{full_model_name}/{model_version}"

client.set_registered_model_alias(
    name=full_model_name, version=model_version, alias="Challenger"
)

print(f"✅ Challenger alias set for version {model_version}")

# COMMAND ----------

# DBTITLE 1,Set model deployment information
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", full_model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
