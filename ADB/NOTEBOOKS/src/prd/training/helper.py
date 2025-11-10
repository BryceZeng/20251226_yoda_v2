from datetime import timedelta, timezone
import math
import yaml
import mlflow.pyfunc
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
import os
from mlflow import MlflowClient


def load_config(env="dev"):
    """
    Load configuration from look-up.yml file.
    Works in both Databricks workspace and local filesystem.
    """
    try:
        # Try Databricks workspace path (when deployed)
        # The workspace root_path is /shared/pac_mlops_new (from databricks.yml)
        try:
            workspace_root = '/Workspace' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/ADB')[0]
            config_path = f"{workspace_root}/Workflows/{env}-commons/look-up.yml"
        except:
            # Fallback: Use relative path from current notebook location
            # From: ADB/NOTEBOOKS/src/prd/training/
            # To:   Workflows/dev-commons/
            # Need to go up 5 levels: training -> prd -> src -> NOTEBOOKS -> ADB -> root
            current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            config_path = os.path.join(current_dir, "../../../../../Workflows", f"{env}-commons", "look-up.yml")
            config_path = os.path.normpath(config_path)
        
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        variables = {k: v.get('default') for k, v in config.get('variables', {}).items()}
        print(f"Loaded {len(variables)} variables from config")
        return variables
        
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        print("Falling back to environment variables and defaults")
        return {}
      

def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_pickup_datetime"], F.lit(15)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_dropoff_datetime"], F.lit(30)
                )
            ),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version