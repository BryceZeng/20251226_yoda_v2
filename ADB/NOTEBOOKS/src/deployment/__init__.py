"""
Databricks Model Serving Deployment Package

This package provides tools for deploying MLflow models from Unity Catalog
to Databricks Model Serving endpoints.

Main Functions:
    deploy_model_endpoint: Deploy or update a model serving endpoint
    get_latest_model_version: Get the latest version of a Unity Catalog model
    get_endpoint_status: Check the status of a serving endpoint
    delete_endpoint: Delete a serving endpoint

Usage:
    from deployment import deploy_model_endpoint

    result = deploy_model_endpoint(
        token=token,
        catalog="analytics_uc",
        schema="ml_models",
        model_name="my_model",
        endpoint_name="my-model-api"
    )

For detailed documentation, see deployment/README.md
"""

from .deploy_endpoint import (  # Backward compatibility
    ModelServingDeploymentError,
    delete_endpoint,
    deploy_model,
    deploy_model_endpoint,
    get_endpoint_status,
    get_latest_model_version,
)

__version__ = "1.0.0"
__author__ = "Analytics Team"

__all__ = [
    "deploy_model_endpoint",
    "get_latest_model_version",
    "get_endpoint_status",
    "delete_endpoint",
    "ModelServingDeploymentError",
    "deploy_model",  # Alias for backward compatibility
]
