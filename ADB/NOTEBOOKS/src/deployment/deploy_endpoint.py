"""
Databricks Model Serving Endpoint Deployment Module

This module provides functionality to deploy MLflow models registered in Unity Catalog
to Databricks Model Serving endpoints. It handles endpoint creation, updates, and
configuration management.

Usage:
    from deployment.deploy_endpoint import deploy_model_endpoint

    deploy_model_endpoint(
        token=dbutils.secrets.get(scope="my-scope", key="databricks-token"),
        catalog="analytics_uc",
        schema="ml_models",
        model_name="my_model",
        endpoint_name="my-model-serving",
        tags={"team": "data-science", "env": "prod"}
    )

Requirements:
    - databricks-sdk>=0.20.0
    - mlflow>=2.9.0
    - pyspark

Author: Analytics Team
Last Updated: December 2025
"""

import logging
import os
from typing import Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelServingDeploymentError(Exception):
    """Custom exception for model serving deployment errors."""
    pass


def _get_databricks_host() -> str:
    """
    Get Databricks workspace URL from Spark configuration.

    Returns:
        str: The Databricks workspace URL

    Raises:
        ModelServingDeploymentError: If workspace URL cannot be determined
    """
    try:
        spark = SparkSession.builder.getOrCreate()
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        return f"https://{workspace_url}"
    except Exception as e:
        raise ModelServingDeploymentError(
            f"Failed to get Databricks workspace URL: {e}"
        )


def _setup_environment(token: str) -> tuple[str, str, str]:
    """
    Set up environment variables required for Databricks API calls.

    Args:
        token: Databricks access token

    Returns:
        Tuple of (token, host, api_base)

    Raises:
        ModelServingDeploymentError: If environment setup fails
    """
    if not token:
        raise ModelServingDeploymentError("Databricks token must be provided")

    try:
        databricks_host = _get_databricks_host()
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["DATABRICKS_HOST"] = databricks_host
        os.environ["DATABRICKS_API_BASE"] = f"{databricks_host}/serving-endpoints/"

        logger.info(f"Environment configured for host: {databricks_host}")
        return token, databricks_host, os.environ["DATABRICKS_API_BASE"]

    except Exception as e:
        raise ModelServingDeploymentError(f"Failed to setup environment: {e}")


def get_latest_model_version(
    catalog: str,
    schema: str,
    model: str,
    client: Optional[MlflowClient] = None
) -> str:
    """
    Fetch the latest model version from Unity Catalog.

    Args:
        catalog: Unity Catalog name
        schema: Schema/database name within the catalog
        model: Registered model name
        client: Optional MLflow client instance

    Returns:
        str: Latest model version number

    Raises:
        ModelServingDeploymentError: If model version cannot be retrieved
    """
    if client is None:
        client = MlflowClient()

    full_model_name = f"{catalog}.{schema}.{model}"
    logger.info(f"Fetching latest version for model: {full_model_name}")

    try:
        versions = client.search_model_versions(f"name='{full_model_name}'")

        if not versions:
            raise ModelServingDeploymentError(
                f"No versions found for model {full_model_name}"
            )

        latest_version = max(versions, key=lambda v: int(v.version)).version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    except Exception as e:
        raise ModelServingDeploymentError(
            f"Error fetching latest version for {full_model_name}: {e}"
        )


def _manage_endpoint_tags(
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    add_tags: Optional[Dict[str, str]] = None,
    delete_tags: Optional[List[str]] = None
) -> None:
    """
    Add or remove tags from an existing endpoint.

    Args:
        workspace_client: Databricks workspace client
        endpoint_name: Name of the endpoint
        add_tags: Dictionary of tags to add/update
        delete_tags: List of tag keys to delete
    """
    try:
        if add_tags:
            endpoint_tags = [EndpointTag(key=k, value=v) for k, v in add_tags.items()]
            workspace_client.serving_endpoints.patch(
                name=endpoint_name,
                add_tags=endpoint_tags
            )
            logger.info(f"Added/updated {len(add_tags)} tags on endpoint {endpoint_name}")

        if delete_tags:
            workspace_client.serving_endpoints.patch(
                name=endpoint_name,
                delete_tags=delete_tags
            )
            logger.info(f"Deleted {len(delete_tags)} tags from endpoint {endpoint_name}")

    except Exception as e:
        logger.warning(f"Failed to manage tags for endpoint {endpoint_name}: {e}")


def deploy_model_endpoint(
    token: str,
    catalog: str,
    schema: str,
    model_name: str,
    endpoint_name: str,
    workload_size: str = "Small",
    scale_to_zero_enabled: bool = True,
    additional_env_vars: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    delete_tags: Optional[List[str]] = None,
    model_version: Optional[str] = None
) -> Dict[str, str]:
    """
    Deploy an MLflow model from Unity Catalog to a Databricks Model Serving endpoint.

    This function will create a new endpoint if it doesn't exist, or update an existing
    endpoint with the latest model version.

    Args:
        token: Databricks access token (use dbutils.secrets.get() in notebooks)
        catalog: Unity Catalog name (e.g., 'analytics_uc')
        schema: Schema/database name within the catalog
        model_name: Name of the registered model
        endpoint_name: Name for the serving endpoint (must be unique)
        workload_size: Workload size - "Small", "Medium", or "Large". Default: "Small"
        scale_to_zero_enabled: Enable auto-scaling to zero. Default: True
        additional_env_vars: Additional environment variables for the endpoint
        tags: Dictionary of tags to add to the endpoint
        delete_tags: List of tag keys to delete from the endpoint
        model_version: Specific model version to deploy (uses latest if None)

    Returns:
        Dict with 'endpoint_url' and 'model_version' keys

    Raises:
        ModelServingDeploymentError: If deployment fails

    Examples:
        >>> # Basic deployment
        >>> result = deploy_model_endpoint(
        ...     token=dbutils.secrets.get("my-scope", "token"),
        ...     catalog="analytics_uc",
        ...     schema="ml_models",
        ...     model_name="customer_churn_model",
        ...     endpoint_name="churn-prediction-api"
        ... )
        >>> print(result['endpoint_url'])

        >>> # Advanced deployment with custom configuration
        >>> result = deploy_model_endpoint(
        ...     token=token,
        ...     catalog="prod_catalog",
        ...     schema="ml",
        ...     model_name="fraud_detector",
        ...     endpoint_name="fraud-detection-v2",
        ...     workload_size="Medium",
        ...     scale_to_zero_enabled=False,
        ...     tags={"team": "fraud-prevention", "env": "production"},
        ...     additional_env_vars={"LOG_LEVEL": "INFO"}
        ... )
    """
    logger.info(f"Starting deployment for model: {catalog}.{schema}.{model_name}")

    # Setup environment
    token, databricks_host, _ = _setup_environment(token)

    # Initialize clients
    mlflow_client = MlflowClient()
    workspace_client = WorkspaceClient()

    # Get model version
    full_model_name = f"{catalog}.{schema}.{model_name}"
    if model_version is None:
        model_version = get_latest_model_version(catalog, schema, model_name, mlflow_client)
    else:
        logger.info(f"Using specified model version: {model_version}")

    # Validate workload size
    valid_sizes = ["Small", "Medium", "Large"]
    if workload_size not in valid_sizes:
        raise ModelServingDeploymentError(
            f"Invalid workload_size '{workload_size}'. Must be one of: {valid_sizes}"
        )

    workload_size_enum = getattr(ServedModelInputWorkloadSize, workload_size.upper())

    # Prepare environment variables
    environment_vars = {
        "DATABRICKS_HOST": databricks_host,
        "DATABRICKS_TOKEN": token
    }
    if additional_env_vars:
        environment_vars.update(additional_env_vars)

    # Configure served model
    served_model = ServedModelInput(
        model_name=full_model_name,
        model_version=model_version,
        workload_size=workload_size_enum,
        scale_to_zero_enabled=scale_to_zero_enabled,
        environment_vars=environment_vars
    )

    endpoint_config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[served_model]
    )

    # Prepare default tags
    default_tags = {
        "Division": "Analytics",
        "Billing": "AI",
        "Nature": "Model-Serving",
        "deployed_by": "deploy_endpoint_module"
    }

    if tags:
        default_tags.update(tags)

    endpoint_tags = [EndpointTag(key=k, value=v) for k, v in default_tags.items()]

    # Check if endpoint exists
    try:
        existing_endpoints = list(workspace_client.serving_endpoints.list())
        existing_endpoint = next(
            (e for e in existing_endpoints if e.name == endpoint_name),
            None
        )
    except Exception as e:
        raise ModelServingDeploymentError(f"Failed to list serving endpoints: {e}")

    # Create or update endpoint
    endpoint_url = f"{databricks_host}/ml/endpoints/{endpoint_name}"

    try:
        if existing_endpoint is None:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            workspace_client.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=endpoint_config,
                tags=endpoint_tags
            )
            logger.info(f"✅ Endpoint created successfully: {endpoint_name}")

        else:
            logger.info(f"Updating existing endpoint: {endpoint_name}")
            workspace_client.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_models=endpoint_config.served_models
            )
            logger.info(f"✅ Endpoint updated successfully: {endpoint_name}")

            # Manage tags for existing endpoint
            _manage_endpoint_tags(workspace_client, endpoint_name, tags, delete_tags)

        logger.info(f"🚀 Endpoint available at: {endpoint_url}")

        return {
            "endpoint_url": endpoint_url,
            "model_version": model_version,
            "full_model_name": full_model_name,
            "endpoint_name": endpoint_name
        }

    except Exception as e:
        raise ModelServingDeploymentError(
            f"Failed to create/update endpoint {endpoint_name}: {e}"
        )


def get_endpoint_status(endpoint_name: str, token: Optional[str] = None) -> Dict:
    """
    Get the current status of a model serving endpoint.

    Args:
        endpoint_name: Name of the endpoint
        token: Databricks access token (optional if already in environment)

    Returns:
        Dict containing endpoint status information

    Raises:
        ModelServingDeploymentError: If status check fails
    """
    try:
        if token:
            _setup_environment(token)

        workspace_client = WorkspaceClient()
        endpoint = workspace_client.serving_endpoints.get(name=endpoint_name)

        return {
            "name": endpoint.name,
            "state": endpoint.state.config_update if endpoint.state else "UNKNOWN",
            "creation_timestamp": endpoint.creation_timestamp,
            "creator": endpoint.creator,
            "tags": {tag.key: tag.value for tag in (endpoint.tags or [])}
        }

    except Exception as e:
        raise ModelServingDeploymentError(
            f"Failed to get status for endpoint {endpoint_name}: {e}"
        )


def delete_endpoint(endpoint_name: str, token: str) -> None:
    """
    Delete a model serving endpoint.

    Args:
        endpoint_name: Name of the endpoint to delete
        token: Databricks access token

    Raises:
        ModelServingDeploymentError: If deletion fails
    """
    logger.warning(f"Attempting to delete endpoint: {endpoint_name}")

    try:
        _setup_environment(token)
        workspace_client = WorkspaceClient()
        workspace_client.serving_endpoints.delete(name=endpoint_name)
        logger.info(f"✅ Endpoint {endpoint_name} deleted successfully")

    except Exception as e:
        raise ModelServingDeploymentError(
            f"Failed to delete endpoint {endpoint_name}: {e}"
        )


# Backward compatibility - keep old function name as alias
deploy_model = deploy_model_endpoint
