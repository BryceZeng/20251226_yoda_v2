# pacs-743-mlops_template

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
> git clone <repository-url>
>
> cd pacs-743-mlops_template

> Install dependencies:
>
> pip install -r requirements.txt

### Components

Azure Databricks Notebooks

> Training Pipeline (model_v1.ipynb): Contains the machine learning model training logic
>
> Inference Pipeline (inference.ipynb): Handles model inference and predictions
>
> Package Configuration (packages.yml): Defines notebook-specific dependencies

Workflow Management

> Environment Configurations: Separate lookup files for development and production environments
>
> Job Templates: Standardized workflow templates for automated job execution

## Development Workflow

pacs-743-mlops_template
├─ ADB
│  ├─ .DS_Store
│  └─ NOTEBOOKS
│     ├─ .DS_Store
│     └─ src
│        ├─ .DS_Store
│        ├─ packages.yml
│        └─ prd
│           ├─ pipelines
│           │  └─ inference.ipynb
│           └─ training
│              └─ model_v1.ipynb
├─ CODEOWNERS
├─ README.md
├─ Workflows
│  ├─ .DS_Store
│  ├─ dev-commons
│  │  └─ look-up.yml
│  ├─ prod-commons
│  │  └─ look-up.yml
│  └─ workflow-yaml
│     └─ Job_Run_<NAME>.yml
├─ databricks.yml
├─ dev-commons
│  └─ look-up.yml
├─ prod-commons
│  └─ look-up.yml
└─ requirements.txt


