# Model Serving Endpoint Deployment

This module provides a clean, production-ready solution for deploying MLflow models registered in Unity Catalog to Databricks Model Serving endpoints.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Integration with Training Pipeline](#integration-with-training-pipeline)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The `deploy_endpoint.py` module simplifies the deployment of trained models to Databricks Model Serving endpoints. It handles:

- ✅ **Unity Catalog Integration**: Works seamlessly with Unity Catalog registered models
- ✅ **Automatic Version Management**: Deploys latest model version or specific versions
- ✅ **Endpoint Lifecycle**: Creates new endpoints or updates existing ones
- ✅ **Tag Management**: Organize endpoints with custom tags
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Status Monitoring**: Check endpoint status and health

## 📦 Installation

### Prerequisites

```bash
databricks-sdk>=0.20.0
mlflow>=2.9.0
pyspark>=3.3.0
```

### Install in Databricks Notebook

```python
%pip install databricks-sdk --upgrade mlflow --quiet
dbutils.library.restartPython()
```

## 🚀 Quick Start

### Basic Deployment

```python
from deployment.deploy_endpoint import deploy_model_endpoint

# Deploy the latest version of your model
result = deploy_model_endpoint(
    token=dbutils.secrets.get("my-scope", "databricks-token"),
    catalog="analytics_uc",
    schema="ml_models",
    model_name="customer_churn_predictor",
    endpoint_name="churn-prediction-api"
)

print(f"✅ Deployed to: {result['endpoint_url']}")
print(f"📦 Model Version: {result['model_version']}")
```

### Advanced Deployment

```python
result = deploy_model_endpoint(
    token=dbutils.secrets.get("my-scope", "databricks-token"),
    catalog="analytics_uc",
    schema="ml_models",
    model_name="fraud_detector",
    endpoint_name="fraud-detection-api",
    workload_size="Medium",           # Small, Medium, or Large
    scale_to_zero_enabled=False,      # Keep endpoint always running
    model_version="5",                # Deploy specific version
    tags={
        "team": "fraud-prevention",
        "environment": "production",
        "sla": "high"
    },
    additional_env_vars={
        "LOG_LEVEL": "INFO",
        "MAX_BATCH_SIZE": "100"
    }
)
```

## 🔗 Integration with Training Pipeline

### Option 1: Add to Training Notebook

Add this at the end of your `TrainWithFeatureStore.py` notebook:

```python
# COMMAND ----------

# DBTITLE 1,Deploy Model to Serving Endpoint (Optional)
# =============================================================================
# Automatic Model Deployment
# =============================================================================
# Automatically deploy the trained model to a serving endpoint.
# Uncomment this section to enable automatic deployment after training.
# =============================================================================

# Import deployment module
import sys
sys.path.append("/Workspace/path/to/ADB/NOTEBOOKS/src")
from deployment.deploy_endpoint import deploy_model_endpoint

# Get configuration
DEPLOY_ENABLED = config.get("deploy_enabled", False)  # Set in lookup.yml

if DEPLOY_ENABLED:
    print("🚀 Starting automatic model deployment...")

    # Get token from secrets
    token = dbutils.secrets.get(
        scope=config.get("secret_scope", "ml-secrets"),
        key="databricks-token"
    )

    # Parse model name to get catalog, schema, model
    model_parts = model_name.split(".")
    if len(model_parts) == 3:
        catalog, schema, model = model_parts

        # Deploy the model
        try:
            result = deploy_model_endpoint(
                token=token,
                catalog=catalog,
                schema=schema,
                model_name=model,
                endpoint_name=config.get("endpoint_name", f"{model}-api"),
                workload_size=config.get("workload_size", "Small"),
                scale_to_zero_enabled=config.get("scale_to_zero", True),
                tags={
                    "model_version": str(latest_version),
                    "training_date": run_timestamp,
                    "project": project_name
                }
            )

            print(f"✅ Model deployed successfully!")
            print(f"📍 Endpoint: {result['endpoint_url']}")

            # Export deployment info for workflow
            dbutils.jobs.taskValues.set("endpoint_url", result['endpoint_url'])
            dbutils.jobs.taskValues.set("deployed_version", result['model_version'])

        except Exception as e:
            print(f"⚠️  Deployment failed (training still succeeded): {e}")
            print("You can deploy manually later using the deployment module")
    else:
        print(f"⚠️  Model name {model_name} not in Unity Catalog format (catalog.schema.model)")
else:
    print("📝 Automatic deployment disabled. Deploy manually if needed.")
```

### Option 2: Separate Deployment Workflow

Create a dedicated deployment task in your workflow YAML:

```yaml
# In Workflows/workflow-yaml/model-workflow-resource.yml

- task_key: deploy_model_to_serving
  depends_on:
    - task_key: model_training
  notebook_task:
    notebook_path: ../src/deployment/deploy_model
    base_parameters:
      catalog: ${var.uc_catalog_name}
      schema: ${var.uc_schema_name}
      model_name: ${var.model_name}
      endpoint_name: ${var.endpoint_name}
      workload_size: "Small"
      scale_to_zero: "true"
  existing_cluster_id: ${var.cluster_id}
```

## 📖 API Reference

### `deploy_model_endpoint()`

Main function to deploy a model to a serving endpoint.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `token` | str | Yes | - | Databricks access token |
| `catalog` | str | Yes | - | Unity Catalog name |
| `schema` | str | Yes | - | Schema within catalog |
| `model_name` | str | Yes | - | Registered model name |
| `endpoint_name` | str | Yes | - | Serving endpoint name |
| `workload_size` | str | No | "Small" | Workload size (Small/Medium/Large) |
| `scale_to_zero_enabled` | bool | No | True | Enable auto-scaling to zero |
| `model_version` | str | No | None | Specific version (latest if None) |
| `additional_env_vars` | dict | No | None | Additional environment variables |
| `tags` | dict | No | None | Tags to add to endpoint |
| `delete_tags` | list | No | None | Tag keys to delete |

**Returns:**

```python
{
    "endpoint_url": "https://workspace.databricks.com/ml/endpoints/my-endpoint",
    "model_version": "3",
    "full_model_name": "catalog.schema.model",
    "endpoint_name": "my-endpoint"
}
```

### `get_endpoint_status()`

Get the current status of a serving endpoint.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `endpoint_name` | str | Yes | Name of endpoint to check |
| `token` | str | No | Token (if not in environment) |

**Returns:**

```python
{
    "name": "my-endpoint",
    "state": "READY",
    "creation_timestamp": 1702858800000,
    "creator": "user@company.com",
    "tags": {"team": "ml", "env": "prod"}
}
```

### `delete_endpoint()`

Delete a serving endpoint.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `endpoint_name` | str | Yes | Name of endpoint to delete |
| `token` | str | Yes | Databricks access token |

## ⚙️ Configuration

### Update lookup.yml

Add deployment configuration to your environment-specific lookup files:

```yaml
# Workflows/dev-commons/look-up.yml

# Model Serving Configuration
deploy_enabled: true
endpoint_name: "customer-churn-api-dev"
workload_size: "Small"
scale_to_zero: true
secret_scope: "ml-dev-secrets"

# Production configuration (Workflows/prod-commons/look-up.yml)
deploy_enabled: true
endpoint_name: "customer-churn-api-prod"
workload_size: "Medium"
scale_to_zero: false
secret_scope: "ml-prod-secrets"
```

### Workload Sizes

| Size | Cores | Memory | Use Case |
|------|-------|--------|----------|
| **Small** | 4 | 16 GB | Development, low traffic |
| **Medium** | 8 | 32 GB | Production, moderate traffic |
| **Large** | 16 | 64 GB | High traffic, large models |

## 🎯 Best Practices

### 1. Token Management

**✅ DO:**
```python
# Store tokens in Databricks secrets
token = dbutils.secrets.get("my-scope", "databricks-token")
```

**❌ DON'T:**
```python
# Never hardcode tokens
token = "dapi1234567890abcdef"  # NEVER DO THIS!
```

### 2. Environment-Specific Configuration

Use different endpoint names and configurations per environment:

```python
# Dev environment
endpoint_name = f"{model_name}-dev"
workload_size = "Small"
scale_to_zero = True

# Production environment
endpoint_name = f"{model_name}-prod"
workload_size = "Medium"
scale_to_zero = False  # Always available
```

### 3. Tagging Strategy

Implement a consistent tagging strategy:

```python
tags = {
    "team": "data-science",
    "project": "customer-retention",
    "environment": "production",
    "model_type": "classification",
    "sla": "high",
    "cost_center": "analytics",
    "deployed_by": "automated-pipeline",
    "deployment_date": datetime.now().isoformat()
}
```

### 4. Error Handling

Always wrap deployment in try-except:

```python
try:
    result = deploy_model_endpoint(...)
    print(f"✅ Deployed: {result['endpoint_url']}")
except ModelServingDeploymentError as e:
    print(f"❌ Deployment failed: {e}")
    # Handle failure (alert, rollback, etc.)
    raise
```

### 5. Version Control

Deploy specific versions for critical production deployments:

```python
# Production: Use specific validated version
result = deploy_model_endpoint(
    ...,
    model_version="5",  # Validated version
    endpoint_name="fraud-detection-prod"
)

# Dev: Use latest version
result = deploy_model_endpoint(
    ...,
    model_version=None,  # Latest
    endpoint_name="fraud-detection-dev"
)
```

## 🔍 Troubleshooting

### Common Issues

#### Issue: "No versions found for model"

**Solution:** Ensure model is registered in Unity Catalog:

```python
import mlflow
mlflow.set_registry_uri("databricks-uc")

# Check if model exists
client = MlflowClient()
try:
    versions = client.search_model_versions("name='catalog.schema.model'")
    print(f"Found {len(versions)} versions")
except:
    print("Model not found - check catalog.schema.model name")
```

#### Issue: "Failed to get Databricks workspace URL"

**Solution:** Ensure running in Databricks environment:

```python
# Check if in Databricks
try:
    spark = SparkSession.builder.getOrCreate()
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    print(f"Workspace: {workspace_url}")
except:
    print("Not in Databricks environment")
```

#### Issue: "Insufficient permissions"

**Solution:** Ensure token has required permissions:
- `CAN MANAGE` on Model Serving
- `CAN READ` on registered model
- `CAN USE CATALOG` and `CAN USE SCHEMA` on Unity Catalog

#### Issue: "Endpoint creation timeout"

**Solution:** Large models may take longer to deploy:

```python
# For large models, use Medium or Large workload size
result = deploy_model_endpoint(
    ...,
    workload_size="Medium",  # More resources
    scale_to_zero_enabled=False  # Keep warm
)
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now deployment will show detailed logs
result = deploy_model_endpoint(...)
```

## 📝 Examples

See [DeploymentExample.ipynb](./DeploymentExample.ipynb) for complete working examples including:

- Basic deployment
- Advanced configuration
- Integration with training pipeline
- Testing deployed endpoints
- Monitoring and status checks

## 🤝 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs with debug logging enabled
3. Contact the ML Platform team

## 📄 License

Internal use only - Analytics Team
