# Testing Strategy & Guidelines

**AI Core:** `Predictive CLTV Insights`
**Industry:** `Media & Entertainment`
**Last Updated:** `2026-02-13`

---

## Table of Contents

1. [Overview](#overview)
2. [Test Strategy](#test-strategy)
3. [Running Tests](#running-tests)
4. [Test Structure](#test-structure)
5. [Coverage Requirements](#coverage-requirements)
6. [Testing Kedro Pipelines](#testing-kedro-pipelines)
7. [Testing Prefect Flows](#testing-prefect-flows)
8. [Mock & Fixture Patterns](#mock--fixture-patterns)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Predictive CLTV Insights AI Core uses **pytest** as the primary testing framework with support for:
- **Unit tests**: Individual node/function testing
- **Integration tests**: Pipeline-level testing with Kedro sessions
- **End-to-end tests**: Full workflow validation (data → models → predictions)
- **Performance tests**: Model training time, inference latency benchmarks

**Testing Philosophy:**
- **Fast feedback loops**: Unit tests < 1 second each
- **Deterministic**: Fixed random seeds (`random_state=42`)
- **Isolated**: No shared state between tests
- **Realistic**: Use representative sample data (1000-10000 rows)

---

## Test Strategy

### Test Pyramid

```
         /\
        /  \  E2E Tests (10%)
       /    \  - Full pipeline runs
      /------\  - Dashboard smoke tests
     /        \ Integration Tests (30%)
    /          \ - Kedro pipeline tests
   /            \ - Feature store tests
  /--------------\ Unit Tests (60%)
 /                \ - Node functions
/_________________ \ - Utility functions
                    - Model validation
```

### Test Levels

| Level | Scope | Tools | Execution Time | Coverage Target |
|-------|-------|-------|----------------|-----------------|
| **Unit** | Individual nodes, utils | pytest | < 1 sec per test | 80% |
| **Integration** | Kedro pipelines | pytest + KedroSession | < 30 sec per pipeline | 70% |
| **E2E** | Full workflow | pytest + Prefect | < 5 min | 50% (critical paths) |
| **Performance** | Model training, inference | pytest-benchmark | Variable | N/A |

### Test Categories by Pipeline

#### Data Processing Pipeline
- **Unit Tests:**
  - Skeleton mapping correctness
  - Data cleaning logic (null handling, negative values)
  - RFM transformation accuracy
  - Feature aggregation logic
- **Integration Tests:**
  - Full data_processing pipeline with sample CSV
  - Delta Lake writes and schema validation
  - Feast feature registration

#### Data Science Pipeline
- **Unit Tests:**
  - BG/NBD model training convergence
  - Gamma-Gamma model parameter validation
  - XGBoost feature importance extraction
  - Prediction confidence interval calculation
- **Integration Tests:**
  - Full data_science pipeline with RFM input
  - MLflow logging and artifact storage
  - Model serialization/deserialization

#### Visualization Pipeline
- **Unit Tests:**
  - Plot generation (PNG file creation)
  - KPI calculation accuracy
  - LLM prompt construction
- **Integration Tests:**
  - Full visualization pipeline with predictions
  - SLM interpretation generation (if available)

---

## Running Tests

### Quick Start

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/test_data_processing.py -v

# Run specific test function
uv run pytest tests/test_data_science.py::test_train_bg_nbd_model -v

# Run tests matching pattern
uv run pytest tests/ -k "test_bg_nbd" -v

# Run tests with specific marker
uv run pytest tests/ -m "unit" -v
uv run pytest tests/ -m "integration" -v

# Run tests in parallel (faster)
uv run pytest tests/ -n auto  # Requires pytest-xdist

# Run tests with detailed output
uv run pytest tests/ -vv --tb=short

# Run tests and stop at first failure
uv run pytest tests/ -x
```

### Coverage Reports

After running tests with `--cov`, open the HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov\index.html  # Windows
```

**Coverage Targets** (configured in `pyproject.toml`):
- **Overall**: 70% line coverage (currently: fail_under=0, should be raised)
- **Critical modules** (nodes.py files): 80%+
- **Utility modules**: 75%+

---

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_data_processing.py        # Data processing pipeline tests
├── test_data_science.py           # Data science pipeline tests
├── test_visualization.py          # Visualization pipeline tests
├── test_utils.py                  # Utility function tests
├── test_core.py                   # Core modules (kedro_pipeline, etc.)
├── fixtures/
│   ├── sample_transactions.csv    # Sample input data
│   ├── sample_rfm.parquet         # Sample RFM data
│   └── sample_predictions.parquet # Sample predictions
└── integration/
    ├── test_e2e_pipeline.py       # End-to-end workflow tests
    └── test_prefect_flows.py      # Prefect orchestration tests
```

### Test Naming Convention

```python
# Unit tests: test_{function_name}
def test_clean_skeleton_data():
    ...

# Integration tests: test_{pipeline_name}_pipeline
def test_data_processing_pipeline():
    ...

# Parametrized tests: test_{function_name}_with_{scenario}
@pytest.mark.parametrize("input_value,expected", [(0, False), (1, True)])
def test_is_churned_with_different_inactivity(input_value, expected):
    ...

# Error tests: test_{function_name}_raises_{exception}
def test_train_bg_nbd_model_raises_value_error():
    ...
```

---

## Coverage Requirements

### Current Coverage Status

Run coverage check:
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Raising Coverage Threshold

**Current** (`pyproject.toml`):
```toml
[tool.coverage.report]
fail_under = 0  # Should be raised to 70
```

**Recommended** (Phase-in approach):
```toml
[tool.coverage.report]
fail_under = 70
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "\\.\\.\\.",  # Ellipsis
]
```

**Coverage Improvement Roadmap:**
1. **Phase 1** (Immediate): Raise `fail_under` to 50%
2. **Phase 2** (1 month): Raise to 65%
3. **Phase 3** (2 months): Raise to 70% (target)

---

## Testing Kedro Pipelines

### Unit Testing Kedro Nodes

```python
import pytest
import polars as pl
from src.ai_core.pipelines.data_processing.nodes import clean_skeleton_data

def test_clean_skeleton_data_removes_nulls():
    """Test that clean_skeleton_data removes rows with null customer_id."""
    # Arrange
    input_data = pl.DataFrame({
        "customer_id": ["C1", None, "C3"],
        "transaction_dt": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "amount_usd": [100.0, 50.0, 75.0]
    })

    # Act
    result = clean_skeleton_data(input_data)

    # Assert
    assert result.height == 2  # Null row removed
    assert "C1" in result["customer_id"]
    assert "C3" in result["customer_id"]

def test_clean_skeleton_data_filters_negative_amounts():
    """Test that clean_skeleton_data removes negative amounts (returns)."""
    # Arrange
    input_data = pl.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "transaction_dt": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "amount_usd": [100.0, -50.0, 0.0]  # Negative and zero
    })

    # Act
    result = clean_skeleton_data(input_data)

    # Assert
    assert result.height == 1  # Only C1 remains
    assert result["customer_id"][0] == "C1"
    assert result["amount_usd"][0] == 100.0
```

### Integration Testing Kedro Pipelines

```python
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

@pytest.fixture(scope="module")
def kedro_project():
    """Bootstrap Kedro project for testing."""
    project_path = Path.cwd()
    bootstrap_project(project_path)
    return project_path

def test_data_processing_pipeline_end_to_end(kedro_project, tmp_path):
    """Test full data_processing pipeline execution."""
    # Arrange: Copy sample data to expected location
    import shutil
    sample_data = Path("tests/fixtures/sample_transactions.csv")
    target = tmp_path / "data/01_raw/online_retail_II.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_data, target)

    # Update catalog.yml to point to tmp_path (or use local overrides)
    # ... (configuration setup)

    # Act: Run pipeline
    with KedroSession.create(project_path=kedro_project) as session:
        context = session.load_context()
        output = session.run(pipeline_name="data_processing")

    # Assert: Check outputs exist
    assert "processed_data" in output  # RFM DataFrame
    assert output["processed_data"].height > 0
    assert "frequency" in output["processed_data"].columns
    assert "recency" in output["processed_data"].columns
    assert "T" in output["processed_data"].columns
```

---

## Testing Prefect Flows

### Mocking Prefect Context

```python
import pytest
from unittest.mock import Mock, patch
from src.prefect_orchestration.data_pipeline import DataPipeline

@pytest.fixture
def mock_prefect_logger():
    """Mock Prefect logger for testing."""
    return Mock()

@patch("src.prefect_orchestration.data_pipeline.get_run_logger")
def test_data_pipeline_setup(mock_get_logger, mock_prefect_logger):
    """Test DataPipeline setup initializes correctly."""
    # Arrange
    mock_get_logger.return_value = mock_prefect_logger
    config = {"project": {"name": "test_project"}, "logging": {"level": "INFO"}}

    # Act
    pipeline = DataPipeline(config)
    pipeline.setup()

    # Assert
    assert pipeline.logger is not None
    mock_prefect_logger.info.assert_called_once()
```

---

## Mock & Fixture Patterns

### Common Fixtures (`conftest.py`)

```python
import pytest
import polars as pl
from pathlib import Path

@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing."""
    return pl.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C3"],
        "transaction_dt": ["2024-01-01", "2024-02-01", "2024-01-15", "2024-03-01"],
        "amount_usd": [100.0, 150.0, 50.0, 200.0]
    })

@pytest.fixture
def sample_rfm_data():
    """Sample RFM data for model training tests."""
    return pl.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "frequency": [3, 0, 1],
        "recency": [30, 0, 15],
        "T": [90, 90, 90],
        "monetary_value": [125.0, 50.0, 100.0]
    })

@pytest.fixture
def trained_bg_nbd_model(sample_rfm_data):
    """Trained BG/NBD model for prediction tests."""
    from src.ai_core.pipelines.data_science.nodes import train_bg_nbd_model
    params = {"penalizer_coef": 0.001}
    return train_bg_nbd_model(sample_rfm_data, params)

@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """Temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")
```

### Mocking MLflow

```python
from unittest.mock import patch, MagicMock

@patch("mlflow.log_metric")
@patch("mlflow.log_param")
def test_model_training_logs_to_mlflow(mock_log_param, mock_log_metric):
    """Test that model training logs parameters and metrics to MLflow."""
    # Arrange
    from src.ai_core.pipelines.data_science.nodes import train_bg_nbd_model
    params = {"penalizer_coef": 0.001}

    # Act
    model = train_bg_nbd_model(sample_rfm_data, params)

    # Assert
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v1

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync

      - name: Run tests with coverage
        run: |
          uv run pytest tests/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-report=term \
            --junitxml=test-results/pytest.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: test-results/**/*.xml
```

---

## Troubleshooting

### Common Issues

#### 1. **Tests Fail with `KedroContextError`**

**Symptom:**
```
KedroContextError: Could not find the project configuration file...
```

**Solution:**
Ensure `bootstrap_project()` is called before creating KedroSession:
```python
from kedro.framework.startup import bootstrap_project
from pathlib import Path

project_path = Path.cwd()
bootstrap_project(project_path)
```

#### 2. **MLflow Tracking Errors in Tests**

**Symptom:**
```
MlflowException: Failed to connect to tracking server
```

**Solution:**
Mock MLflow in unit tests or use `mlflow.start_run(nested=True)` in integration tests:
```python
@patch("mlflow.start_run")
def test_with_mlflow_mocked(mock_start_run):
    ...
```

#### 3. **Feast Feature Store Not Found**

**Symptom:**
```
FeatureStoreNotFound: No feature store found at ...
```

**Solution:**
Skip Feast-dependent tests if feature store unavailable:
```python
import pytest

@pytest.mark.skipif(
    not Path("feature_repo/feature_store.yaml").exists(),
    reason="Feast feature store not configured"
)
def test_feast_materialization():
    ...
```

#### 4. **Random Test Failures (Flakiness)**

**Symptom:**
Tests pass/fail inconsistently due to non-determinism.

**Solution:**
Fix random seeds in tests:
```python
import numpy as np
import random

@pytest.fixture(autouse=True)
def fix_random_seeds():
    """Fix random seeds for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    # For torch/TensorFlow:
    # torch.manual_seed(42)
```

#### 5. **Slow Test Execution**

**Symptom:**
Test suite takes > 5 minutes to run.

**Solution:**
- Run tests in parallel: `uv run pytest tests/ -n auto`
- Use smaller sample datasets (< 1000 rows)
- Mark slow tests with `@pytest.mark.slow` and skip: `pytest -m "not slow"`

---

## Test Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Coverage meets threshold: `uv run pytest tests/ --cov=src --cov-report=term`
- [ ] New code has corresponding tests (unit + integration)
- [ ] Tests are deterministic (fixed random seeds)
- [ ] No test warnings or deprecation notices
- [ ] Integration tests pass in CI/CD pipeline
- [ ] Code follows testing conventions (naming, structure)

---

**Document Control:**
- **Review Cycle:** Quarterly or on major feature additions
- **Approval Required:** Tech Lead, QA Lead
- **Related Documents:** `technical_design.md`, `CONTRIBUTING.md`
