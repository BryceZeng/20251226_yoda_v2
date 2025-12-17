# 🚀 Quick Start Guide - Model Serving Deployment

**Goal:** Deploy your trained model to a Databricks serving endpoint in 5 minutes.

## ✅ Prerequisites

- [ ] Model registered in Unity Catalog (format: `catalog.schema.model_name`)
- [ ] Databricks token stored in secrets
- [ ] Python 3.8+ with databricks-sdk and mlflow installed

## 📝 Step 1: Store Your Databricks Token

```bash
# Using Databricks CLI
databricks secrets create-scope --scope ml-secrets
databricks secrets put --scope ml-secrets --key databricks-token
# Paste your token when prompted
```

## 📦 Step 2: Install Required Packages

In your Databricks notebook:

```python
%pip install databricks-sdk --upgrade mlflow --quiet
dbutils.library.restartPython()
```

## 🎯 Step 3: Deploy Your Model

### Option A: Quick Deploy (3 lines)

```python
from deployment.deploy_endpoint import deploy_model_endpoint

result = deploy_model_endpoint(
    token=dbutils.secrets.get("ml-secrets", "databricks-token"),
    catalog="analytics_uc",           # Your catalog
    schema="ml_models",                # Your schema
    model_name="my_model",            # Your model name
    endpoint_name="my-model-api"      # Desired endpoint name
)

print(f"✅ Deployed! Access at: {result['endpoint_url']}")
```

### Option B: Production Deploy (with configuration)

```python
from deployment.deploy_endpoint import deploy_model_endpoint

result = deploy_model_endpoint(
    token=dbutils.secrets.get("ml-prod-secrets", "databricks-token"),
    catalog="prod_catalog",
    schema="ml_models",
    model_name="fraud_detector",
    endpoint_name="fraud-detection-prod",
    workload_size="Medium",           # Small/Medium/Large
    scale_to_zero_enabled=False,      # Keep always running
    tags={
        "team": "fraud-prevention",
        "environment": "production",
        "sla": "high"
    }
)

print(f"✅ Production endpoint ready: {result['endpoint_url']}")
```

## ✨ Step 4: Test Your Endpoint

```python
import requests
import json

# Get workspace URL
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f"https://{workspace_url}/serving-endpoints/{result['endpoint_name']}/invocations"

# Prepare test data (adjust to your model's schema)
test_data = {
    "dataframe_records": [
        {
            "feature1": 1.0,
            "feature2": 2.5,
            "feature3": "category_a"
        }
    ]
}

# Make prediction
headers = {
    "Authorization": f"Bearer {dbutils.secrets.get('ml-secrets', 'databricks-token')}",
    "Content-Type": "application/json"
}

response = requests.post(endpoint_url, headers=headers, json=test_data)

if response.status_code == 200:
    print("✅ Prediction successful!")
    print(f"Result: {response.json()}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
```

## 🎓 Common Use Cases

### Use Case 1: Add to Training Notebook

At the end of your training script:

```python
# ... your training code ...

# Register model
import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.sklearn.log_model(model, "model", 
                        registered_model_name=f"{catalog}.{schema}.{model_name}")

# Deploy immediately
from deployment.deploy_endpoint import deploy_model_endpoint

result = deploy_model_endpoint(
    token=dbutils.secrets.get("ml-secrets", "databricks-token"),
    catalog=catalog,
    schema=schema,
    model_name=model_name,
    endpoint_name=f"{model_name}-api"
)

print(f"🎉 Model trained and deployed: {result['endpoint_url']}")
```

### Use Case 2: Deploy from Workflow

Create a workflow task using `deploy_model.ipynb`:

```yaml
# workflow-yaml/model-workflow-resource.yml
tasks:
  - task_key: train_model
    notebook_task:
      notebook_path: ../src/training/TrainWithFeatureStore.py
    
  - task_key: deploy_model
    depends_on:
      - task_key: train_model
    notebook_task:
      notebook_path: ../src/deployment/deploy_model
      base_parameters:
        catalog: "analytics_uc"
        schema: "ml_models"
        model_name: "my_model"
        endpoint_name: "my-model-api-prod"
        workload_size: "Medium"
        secret_scope: "ml-prod-secrets"
```

### Use Case 3: Update Existing Endpoint

```python
# Deploy new version to existing endpoint
from deployment.deploy_endpoint import deploy_model_endpoint

# This will update the endpoint with the latest model version
result = deploy_model_endpoint(
    token=dbutils.secrets.get("ml-secrets", "databricks-token"),
    catalog="analytics_uc",
    schema="ml_models",
    model_name="my_model",
    endpoint_name="existing-endpoint-name"  # Same name = update
)

print(f"✅ Endpoint updated to version {result['model_version']}")
```

## 🔍 Monitoring

### Check Endpoint Status

```python
from deployment.deploy_endpoint import get_endpoint_status

status = get_endpoint_status("my-model-api", token)
print(f"State: {status['state']}")
print(f"Creator: {status['creator']}")
```

### View in Databricks UI

1. Go to **Machine Learning** → **Serving**
2. Find your endpoint by name
3. View metrics, logs, and configuration

## ⚙️ Configuration by Environment

### Development
```python
endpoint_config = {
    "endpoint_name": f"{model_name}-dev",
    "workload_size": "Small",
    "scale_to_zero_enabled": True,
    "tags": {"env": "dev"}
}
```

### Production
```python
endpoint_config = {
    "endpoint_name": f"{model_name}-prod",
    "workload_size": "Medium",
    "scale_to_zero_enabled": False,  # Always available
    "tags": {"env": "prod", "sla": "high"}
}
```

## 🆘 Troubleshooting

### Error: "No versions found for model"

**Problem:** Model not in Unity Catalog format

**Solution:**
```python
# Check model exists
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Search for model
versions = client.search_model_versions("name='catalog.schema.model'")
print(f"Found {len(versions)} versions")

# If none found, register model first
import mlflow
mlflow.set_registry_uri("databricks-uc")
# ... register your model
```

### Error: "Failed to get Databricks workspace URL"

**Problem:** Not running in Databricks environment

**Solution:** This module must run in Databricks notebooks/jobs

### Error: "Insufficient permissions"

**Problem:** Token lacks required permissions

**Solution:** Ensure token has:
- `CAN MANAGE` on Model Serving
- `CAN READ` on registered model
- `CAN USE CATALOG` and `CAN USE SCHEMA`

### Endpoint takes long to create

**Solution:** Normal for first deployment. Large models may take 5-10 minutes.

```python
# Check status while waiting
from deployment.deploy_endpoint import get_endpoint_status
status = get_endpoint_status("my-endpoint", token)
print(f"Current state: {status['state']}")
```

## 📚 Next Steps

1. **Read Full Documentation:** [README.md](./README.md)
2. **Try Examples:** [DeploymentExample.ipynb](./DeploymentExample.ipynb)
3. **Review Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)
4. **Integration Guide:** [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

## 💡 Pro Tips

1. **Use separate endpoints for dev/prod**
   ```python
   endpoint_name = f"{model_name}-{environment}"  # e.g., "churn-model-prod"
   ```

2. **Tag everything for cost tracking**
   ```python
   tags = {
       "team": "data-science",
       "project": "customer-retention",
       "cost_center": "analytics"
   }
   ```

3. **Test in dev first**
   ```python
   # Deploy to dev, run tests, then deploy to prod
   dev_result = deploy_model_endpoint(..., endpoint_name="model-dev")
   # ... run tests ...
   prod_result = deploy_model_endpoint(..., endpoint_name="model-prod")
   ```

4. **Pin specific versions in production**
   ```python
   result = deploy_model_endpoint(
       ...,
       model_version="5",  # Validated version
       endpoint_name="model-prod"
   )
   ```

5. **Monitor your endpoints**
   - Set up alerts for endpoint failures
   - Monitor latency and throughput
   - Track prediction quality

## 🎯 Success Checklist

- [ ] Token stored in Databricks secrets
- [ ] Model registered in Unity Catalog format
- [ ] Deployment successful
- [ ] Endpoint tested with sample data
- [ ] Endpoint URL shared with application team
- [ ] Monitoring configured
- [ ] Documentation updated

---

**Questions?** Check [README.md](./README.md) or contact the ML Platform team.

**Version:** 1.0.0 | **Last Updated:** December 17, 2025
