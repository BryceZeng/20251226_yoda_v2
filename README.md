# pacs-743-model_deployment_template Companion Guide

This document complements the main `README.md` and focuses on three topics:

repo layout, how to clone/bootstrap the project, and guidance for teams that

want to copy this repository as an MLOps template.

### Getting Started

Prerequisites

> Azure Databricks workspace access
>
> Databricks CLI installed and configured
>
> Required permissions as defined in CODEOWNERS

Setup

> Clone the repository:
>
> Example
>
> git clone `<repository-url>`
>
> cd pacs-743-mlops_template

> Install dependencies:
>
> pip install -r requirements.txt

## Repository Structure

The repository lives under `pacs-743-model_deployment_template/` (this folder)

inside the workspace. The tree below lists every tracked folder and file along

with its purpose so teams can quickly navigate or replicate the scaffolding.

```
pacs-743-model_deployment_template/
├─ .gitignore                      # Ignore rules for Git + Databricks bundles
├─ @README.md                      # Companion guide (this document)
├─ CODEOWNERS                      # Ownership + review gates
├─ README.md                       # Primary quick-start instructions
├─ databricks.yml                  # Databricks bundle definition
├─ requirements.txt                # Python dependency lock for local runs
├─ Workflows/                      # Job templates and environment lookups
│  ├─ dev-commons/                 # Dev environment parameter lookup
│  │  └─ look-up.yml               # Dev job settings (clusters, paths, secrets)
│  ├─ prod-commons/                # Prod environment parameter lookup
│  │  └─ look-up.yml               # Prod job settings (clusters, paths, secrets)
│  ├─ uat-commons/                 # UAT environment parameter lookup
│  │  └─ look-up.yml               # UAT job settings (clusters, paths, secrets)
│  ├─ test/                        # Templates for validating workflow configs
│  │  └─ unit-test-workflow-resource.yml
│  └─ workflow-yaml/               # Canonical Databricks job specs
│     ├─ batch-inference-workflow-resource.yml   # Batch inference DAG
│     ├─ feature-engineering-workflow-resource.yml # Feature build DAG
│     ├─ model-workflow-resource.yml             # Training job definition
│     └─ promote-model-resource.yml              # Model promotion workflow
├─ .history/                       # Local change snapshots (Cursor/VS Code)
│  └─ CODEOWNERS_20251031154736    # Example tracked historical copy
├─ ADB/                            # Databricks-notebook source of truth
│  └─ NOTEBOOKS/
│     └─ src/
│        ├─ packages.yml           # Notebook-scoped Python dependencies
│        ├─ deployment/            # Deployment scripts and configurations
│        ├─ promote/
│        │  └─ promote_model.ipynb # Model registry promotion notebook
│        ├─ training/
│        │  ├─ TrainWithFeatureStore.py # End-to-end training pipeline
│        │  └─ helper.py                 # Shared training utilities
│        ├─ tests/
│        │  ├─ __init__.py               # Marks module for pytest/databricks
│        │  └─ unit_tests.py             # Unit tests for notebook logic
│        ├─ inference/
│        │  ├─ BatchInference.py         # Batch inference orchestration
│        │  └─ predict.py                # Prediction entry point
│        ├─ feature_engineering/
│        │  ├─ __init__.py               # Package marker
│        │  ├─ features/
│        │  │  ├─ __init__.py            # Feature module marker
│        │  │  └─ simple_features.py     # Main feature generation logic
│        │  └─ notebooks/
│        │     ├─ CreateGroundTruth.ipynb         # Ground truth generation notebook
│        │     ├─ GenerateAndWriteFeatures.ipynb # Feature generation notebook
│        │     └─ GenerateAndWriteFeatures.py     # Feature generation script
│        └─ validation/
│           └─ challenger_validation.py  # Challenger vs champion validation
└─ .github/
   └─ workflows/
      └─ trigger-deploy.yml                 # Automated deployment workflow
```

> Tip: keep the path casing and hierarchy intact when cloning or templating so
>
> workflow references and bundle manifests remain valid.

## GitHub Actions Workflows

Located in `.github/workflows/`, this pipeline automates Databricks bundle

validation, deployment, and promotions through a reusable workflow architecture.

When cloning this repository for other teams, review the workflow so triggers,

repo references, and secrets align with the new environment.

- `pacs-datahub-ai-deploy-databricks-asset-bundle.yml`
    - **Purpose**: CI deploy pipeline for Databricks Asset Bundles (DAB). Validates

      the bundle, runs tests, and (optionally) deploys to a target workspace after

      merges.
    - **Key sections**:
        - `on`: trigger configuration (push to main branches, pull requests,

          `workflow_dispatch`).
        - `env` / `with`: workspace configuration (workspace URL, bundle target,

          bundle root).
        - `secrets`: expects Databricks PAT and optional Azure credentials.
    - **How to adapt**:

      1. Change branch filters in the `on` block to match your default branch.
      2. Update bundle names, workspace IDs, and paths under `env`.
      3. Ensure referenced CLI commands point to the right `databricks.yml`.
      4. Create the required GitHub secrets (`DATABRICKS_TOKEN`, etc.) in the new

         repository or rename them consistently.
- `trigger-deploy.yml`
    - **Purpose**: Automated deployment workflow that triggers on branch pushes to deploy Databricks asset bundles to appropriate environments.
    - **Key sections**:
        - `on`: triggered by pushes to `master`, `develop`, and `feature/bryce_v4` branches.
        - Three conditional jobs:
            - `call-master`: Deploys to prod environment when pushing to master, runs promote_model_job
            - `call-develop`: Deploys to dev environment when pushing to develop, runs model_training_job and unit tests
            - `call-feature`: Deploys to dev environment when pushing to feature/<name>, runs model_training_job and unit tests
        - Uses reusable workflow from `pru-pss/pss-eta-reusable_workflows` repository
    - **How to adapt**:

      1. Update branch filters in the `on` block to match your branch strategy.
      2. Modify the conditional jobs (`call-master`, `call-develop`, `call-feature`) to align with your environments.
      3. Update environment names, runner names, and job names in the `with` parameters.
      4. Ensure required GitHub secrets exist: `PERSONAL_PRD2_ACCESS_TOKEN`, `PERSONAL_DEV3_ACCESS_TOKEN`, `DATABRICKS_PRD2_HOST`, `DATABRICKS_DEV3_HOST`, `GIT_TOKEN`.
      5. Update the reusable workflow reference if using a different organization or workflow.

> Tip: keep workflow filenames stable if other governance automation references
>
> them; otherwise update downstream dashboards or composite workflows as well.

## Environment Configuration Files

Environment-specific knobs live under `Workflows/dev-commons/`, `Workflows/uat-commons/`, and

`Workflows/prod-commons/`. Each `look-up.yml` maps bundle targets to cluster IDs,

job parameters, secret scopes, and storage paths.

- **Required keys**: `workspace_url`, `cluster_id` or `new_cluster`,

  `job_parameters`, `artifact_path`, and any `secret_scope` references used

  inside notebooks.
- **Local testing tip**: copy the dev lookup file, adjust paths for your sandbox

  workspace, and reference it via `databricks bundle run --target dev`.
- **Secret management**: prefer Databricks secret scopes and reference them in

  lookup files rather than embedding values directly.

## Databricks Bundle Lifecycle

1. `databricks bundle validate` – ensures `databricks.yml` and workflows are

   syntactically correct. Run locally and in CI.
1. `databricks bundle deploy --target <env>` – uploads resources, registers jobs,

   and syncs notebooks. Targets map to entries in `databricks.yml`.
1. `databricks bundle run --target <env> <workflow>` – triggers a specific job

   (e.g., `model-workflow`). Use after deploy or from GitHub workflows.
1. Promotion – once validated in dev, re-run deploy/run against prod targets

   using the `trigger-deploy` workflow or the CLI.
1. Rollback – re-run deploy with a previous git ref; bundle metadata keeps job

   versions aligned.

## Testing and Quality Gates

- `ADB/NOTEBOOKS/src/tests/unit_tests.py` contains pytest-compatible tests

  for helper logic. Run `pytest ADB/NOTEBOOKS/src/tests` locally.
- Extend tests alongside new notebooks or utilities; GitHub Actions can run them

  before bundle deployment by adding a test step.
- Consider adding data validation tests in `validation/` to assert schema or

  metric constraints prior to promotion.

## Promotion Strategy

- `promote/promote_model.ipynb` orchestrates MLflow model registry promotion.
- Recommended flow: run challenger validation, confirm metrics, execute the

  promotion notebook with the target stage (e.g., Staging → Production), then run

  inference workflows to build new artifacts.
- Ensure the service principal or PAT has `CAN_MANAGE` rights on the MLflow

  registry; otherwise promotions will fail.

## ML Engineer Playbook

- **Feature store + data lineage**: Keep feature definitions in `feature_engineering/features/` aligned with Delta tables or the Databricks Feature Store. Document upstream sources (catalog.schema.table) in module docstrings and internal wikis so other teams can trace inputs. Register new tables via Unity Catalog and use fully qualified names in notebooks to avoid ambiguity.
- **Model experimentation guidance**: Clone `TrainWithFeatureStore.py` into a personal branch or workspace folder, log hyperparameters/metrics to MLflow with commit hashes as tags, and use MLflow's compare view to pick the best run before merging back.
- **Performance/tuning tips**: Start with Photon-enabled clusters for ETL, switch to GPU runtimes when workloads demand it, and capture cluster specs in lookup files for reproducibility. Profile slow stages in the Spark UI and leverage caching, Delta optimize write, and adaptive query execution.
- **Deployment playbooks**: Maintain champion/challenger models in the registry, promote challengers to Staging, execute batch inference for smoke tests, then advance to Production when KPIs pass. Roll back by re-promoting the previous MLflow version and rerunning inference jobs.
- **Observability hooks**: Emit latency, accuracy, and drift metrics via `mlflow.log_metric` in prediction scripts, forward structured logs to Lakehouse Monitoring or Prometheus helpers under `validation/`, and schedule validation workflows to alert when thresholds breach.

## Platform Engineer Checklist

- **Infrastructure baselines**: Define cluster policies (node types, autoscaling, DBR versions) and reference them from `Workflows/*/look-up.yml`. Enforce policy IDs in Databricks so every job launched from this template inherits guardrails.
- **Security & compliance**: Require Unity Catalog-enabled workspaces, ensure secret scopes hold credentials (never inline), and set PAT/service principal rotation cadences. Verify GitHub Actions runners use OIDC or short-lived tokens.
- **CI/CD governance**: Align workflows with enterprise reusable workflows or require approvals on protected branches. Wire required status checks to the bundle deploy workflow and store artifacts/logs centrally for audits.
- **Cost & quota management**: Monitor job usage via Databricks cost dashboards, set default timeouts/retries in lookup files, and consider auto-termination settings to avoid idle clusters.
- **Observability & logging**: Forward Databricks job logs to the org-wide logging stack (Splunk, Azure Monitor, etc.) and configure webhook alerts for failed jobs so platform on-call can triage quickly.
- **Template lifecycle**: Tag releases of this repository, publish changelogs, and provide migration guidance when updating bundle schemas or workflow contracts.

## Customization Checklist

| Area         | Files                                                | Typical changes                                                                        |
| ------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Naming       | `Workflows/workflow-yaml/*.yml`, `databricks.yml`    | Rename jobs, targets, and bundle names to match new team/project.                      |
| Data sources | `feature_engineering/features/*.py`, `training/*.py` | Point to new tables, feature store locations, and ML tasks.                            |
| Clusters     | `Workflows/*/look-up.yml`, `databricks.yml`          | Update cluster specs, node types, or DBR versions for dev, uat, and prod environments. |
| Secrets      | `.github/workflows/*.yml`, lookup files              | Configure GitHub secrets and Databricks secret scopes referenced in code.              |
| Tests        | `tests/unit_tests.py`, `validation/*.py`             | Add coverage for new logic and validation rules.                                       |

## Operational Runbook

- **CLI auth errors**: rerun `databricks configure --token` or refresh the PAT

  used by GitHub secrets.
- **Workflow failure debugging**: check the Databricks job run output via the

  workspace UI; cross-reference workflow run IDs emitted in GitHub Actions.
- **Bundle drift**: if manual notebook edits occur in the workspace, re-run

  `databricks bundle deploy` to resync from Git.
- **Secret mismatches**: confirm secret names align across GitHub, lookup files,

  and notebooks; mismatches surface as `KeyError` or auth failures.
- **Resource cleanup**: periodically prune unused workflows or models via the

  Databricks UI to keep bundle deploys fast.

## Clone and Bootstrap

```bash
git clone <repository-url>
cd pacs-743-model_deployment_template
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
databricks bundle validate   # Optional, ensures CLI sees the project
```

Prerequisites:

1. Access to the target Azure Databricks workspace.
2. Databricks CLI v0.205+ installed and configured (`databricks configure --token`).
3. Permissions listed in `CODEOWNERS`.

## Using This Repo as an MLOps Template

1. **Fork or scaffold** – Copy this repository (or use GitHub’s “Use this

   template” button if available) into your team’s namespace.
1. **Rename logical namespaces** – Update workflow names and notebook folders

   under `ADB/NOTEBOOKS/src/` to match your model or domain nomenclature.
1. **Swap data sources** – Modify feature notebooks in

   `feature_engineering/features/` and training scripts in `training/` to point

   to your data stores, feature tables, and ML tasks.
1. **Update bundles** – Edit `databricks.yml` with your workspace IDs, clusters,

   and Git repo paths. Regenerate any secrets or tokens referenced in workflow

   lookups (`Workflows/*/look-up.yml`).
1. **Regenerate workflows** – Duplicate or edit the YAML specs in

   `Workflows/workflow-yaml/` to reflect your pipelines (feature engineering,

   training, inference, promotion). Use the existing files as canonical

   templates for job parameters, task chaining, and artifact promotion.
1. **Validate end-to-end** – Run local unit tests (`ADB/NOTEBOOKS/src/tests`)

   and execute Databricks bundle deploy + run commands in dev before promoting

   to prod.

Following these steps lets any team bootstrap a full MLOps workflow—feature

engineering, model training, batch inference, validation, and promotion—while

retaining the guardrails and structure proven in this template.