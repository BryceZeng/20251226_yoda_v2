# Parameter Configuration Validation Checklist

## Environment Setup Validation

### 1. Required Files
- [ ] `/Workflows/dev-commons/look-up.yml` exists and is properly configured
- [ ] `/Workflows/uat-commons/look-up.yml` exists and is properly configured
- [ ] `/Workflows/prod-commons/look-up.yml` exists and is properly configured
- [ ] `databricks.yml` includes environment-specific commons files

### 2. Core Parameters (All Environments)
- [ ] TASK_KEY is defined with description
- [ ] SCHEMA is defined with description
- [ ] CATALOG is environment-appropriate (pru/pru_uat/pru_prod)
- [ ] TRAINING_DATA_PATH matches catalog environment
- [ ] MODEL_NAME is consistent across environments
- [ ] DEST_MODEL_NAME matches target environment

### 3. Feature Store Parameters
- [ ] PICKUP_FEATURES_TABLE uses correct catalog prefix
- [ ] DROP_FEATURES_TABLE uses correct catalog prefix
- [ ] Tables are consistently named across environments

### 4. Inference Parameters
- [ ] INFERENCE_INPUT_TABLE uses correct catalog prefix
- [ ] OUTPUT_PREDICTION_TABLE uses correct catalog prefix
- [ ] GROUND_TRUTH_TABLE uses correct catalog prefix

### 5. Git Configuration
- [ ] GIT_URL is appropriate for each environment
- [ ] GIT_BRANCH matches environment (feature/* for dev, release/* for uat, main for prod)
- [ ] ENVIRONMENT variable matches target environment

### 6. Compute Resources
- [ ] cluster_id is defined (static ID or lookup)
- [ ] WAREHOUSE_ID uses appropriate warehouse for environment
- [ ] Cluster configurations are appropriate for environment load

### 7. Workflow Integration
- [ ] All variables used in workflow files are defined in look-up files
- [ ] Variable names are consistent between look-up and workflow files
- [ ] No hardcoded values remain in workflow files

## Environment-Specific Validation

### Development Environment
- [ ] Uses development catalog (pru)
- [ ] Uses feature branch for Git
- [ ] Has appropriate cluster sizing for development work

### UAT Environment
- [ ] Uses UAT catalog (pru_uat)
- [ ] Uses release branch for Git
- [ ] Has testing-appropriate configurations
- [ ] Includes UAT-specific tags and metadata

### Production Environment
- [ ] Uses production catalog (pru_prod)
- [ ] Uses main/master branch for Git
- [ ] Has production-grade cluster configurations
- [ ] Includes production monitoring and notification settings

## Parameter Robustness Checks

### 1. Documentation
- [ ] All parameters have meaningful descriptions
- [ ] Complex parameters have type specifications
- [ ] Environment differences are clearly documented

### 2. Default Values
- [ ] All parameters have sensible default values
- [ ] No placeholder/dummy values remain
- [ ] Environment-specific defaults are appropriate

### 3. Variable References
- [ ] Variable substitutions use correct syntax: ${var.VARIABLE_NAME}
- [ ] No circular references between variables
- [ ] Dependent variables are properly ordered

### 4. Security Considerations
- [ ] No sensitive values are hardcoded in look-up files
- [ ] Cluster IDs and resource references are appropriate for security model
- [ ] Git URLs and branches follow security policies

## Testing and Validation Commands

After making changes, validate with:

```bash
# Validate YAML syntax
databricks bundle validate --target dev
databricks bundle validate --target uat
databricks bundle validate --target prod

# Deploy to development for testing
databricks bundle deploy --target dev

# Check deployed resources
databricks workspace list /Workspace/...
databricks jobs list
```

## Common Issues and Solutions

### Missing Variable Error
- **Issue**: `Error: variable "VARIABLE_NAME" is not defined`
- **Solution**: Add the variable to the appropriate environment's look-up.yml file

### Wrong Environment Values
- **Issue**: Development resources appearing in production
- **Solution**: Verify catalog prefixes and environment-specific values in look-up files

### Cluster Access Issues
- **Issue**: Jobs failing due to cluster access
- **Solution**: Verify cluster_id values and user permissions

### Git Integration Problems
- **Issue**: Wrong branch or repository in deployments
- **Solution**: Check GIT_URL and GIT_BRANCH values in environment-specific look-up files