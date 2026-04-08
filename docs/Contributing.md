# Contributing Guide: Adding a New AI Core

Welcome! This guide walks you through the process of creating and contributing a new AI Core to the monorepo.

## 1. Prerequisites

- **uv** installed.
- Access to the `ai_core_template`.
- Basic understanding of **Kedro** and **Prefect**.

## 2. Scaffolding Your AI Core

We provide a template and a script to automate the creation of a new core.

### Step 1: Run the Creation Script
From the root of the repository, run the creation script (adjusting paths as necessary):

```bash
uv run python cores/industry_template/ai_core_template/scripts/create_new_core.py 
  --name <your-core-name> 
  --industry <industry-folder> 
  --output-dir cores/
```

### Step 2: Initialize the Environment
Navigate to your new core and sync the dependencies:

```bash
cd cores/<industry>/<your-core-name>
uv sync --extra mps  # Use cuda or xpu if on Linux/Windows
```

## 3. Implementation Checklist

Every AI Core must implement the following standardized components:

### 📝 Documentation
Update the following files in your core's `docs/` folder:
- `ai_product_canvas.md`: Business logic and KPIs.
- `technical_design.md`: Architecture decisions.
- `root_context.md`: Operating modes for the AI agent.

### ⚙️ Configuration
- **Catalog**: Define datasets in `conf/base/catalog.yml`.
- **Parameters**: Define hyperparameters in `conf/base/parameters.yml`.
- **Orchestration**: Configure Prefect flows in `configs/project_config.yaml`.

### 🧪 Pipelines
- Implement transformation nodes in `src/<package_name>/pipelines/data_processing`.
- Implement model nodes in `src/<package_name>/pipelines/data_science`.
- Register everything in `src/<package_name>/pipeline_registry.py`.

## 4. Testing Your Changes

### Unit Tests
```bash
uv run pytest tests/unit/
```

### Dry Run Pipeline
```bash
uv run kedro run --pipeline data_processing
```

## 5. Pull Request Process

1. **Branching**: Create a feature branch (e.g., `feat/churn-forecasting`).
2. **Linting**: Run `ruff check .` or your project's linting tool.
3. **Documentation**: Ensure the root `README.md` is updated to include your new core in the index.
4. **Review**: Submit the PR and wait for review from the AI & Data Engineering team.

## 6. Coding Standards

Follow the rules defined in [Standards.md](./Standards.md). Most importantly:
- Use **Type Hints** everywhere.
- Follow **Google Style** docstrings.
- Adhere to the **Pickle Rule**.
