# Contributing to Predictive CLTV Insights

Thank you for your interest in contributing to the **Predictive CLTV Insights** AI Core! This guide will help you get started with contributing code, documentation, and ideas to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Standards](#documentation-standards)
7. [Pull Request Process](#pull-request-process)
8. [Adding New Features](#adding-new-features)
9. [Adding New Models](#adding-new-models)
10. [Troubleshooting](#troubleshooting)

---

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming and inclusive environment:

- **Be respectful**: Treat all contributors with respect and professionalism
- **Be collaborative**: Work together to solve problems and improve the project
- **Be constructive**: Provide helpful feedback and accept criticism gracefully
- **Be inclusive**: Welcome contributors of all backgrounds and skill levels

Violations can be reported to the project maintainers.

---

## Getting Started

### Prerequisites

- **Python 3.12** (>=3.11, <3.13)
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **Git** for version control
- Recommended: Hardware acceleration (GPU/XPU/MPS) for model training

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai_cores.git
cd ai_cores/cores/media_entertainment/predictive-CLTV-insights

# Add upstream remote
git remote add upstream https://github.com/Wizeline/ai_cores.git
```

### Install Dependencies

```bash
# Install core dependencies
uv sync

# For Apple Silicon (MPS acceleration)
uv sync --extra mps

# For NVIDIA GPUs (CUDA acceleration)
uv sync --extra cuda

# For Intel Arc GPUs (XPU acceleration)
uv sync --extra xpu
```

### Verify Setup

```bash
# Run tests to ensure everything works
uv run pytest tests/ -v

# Check code style
uv run black --check src/
uv run flake8 src/

# Run a sample pipeline
uv run kedro run --pipeline=data_processing
```

---

## Development Workflow

### Branch Strategy

We use **Git Flow** with the following branches:

- `main`: Production-ready code (protected)
- `develop`: Integration branch for features (protected)
- `feature/*`: New features (e.g., `feature/add-deepsurvival-model`)
- `bugfix/*`: Bug fixes (e.g., `bugfix/fix-feast-materialization`)
- `hotfix/*`: Urgent production fixes (e.g., `hotfix/critical-prediction-error`)

### Creating a Feature Branch

```bash
# Update develop branch
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: Add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Convention

We follow **Conventional Commits** for clear history:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no feature/bug change)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, config)

**Examples:**
```bash
git commit -m "feat(data_science): Add DeepSurv survival model"
git commit -m "fix(feast): Resolve materialization timeout"
git commit -m "docs(api): Add Python client examples"
git commit -m "test(data_processing): Add RFM transformation tests"
```

---

## Code Style Guidelines

### Python Style

We enforce code style using **Black**, **Flake8**, and **mypy**:

```bash
# Format code with Black
uv run black src/

# Check linting with Flake8
uv run flake8 src/

# Type check with mypy
uv run mypy src/
```

**Configuration** (in `pyproject.toml`):
- Line length: 100 characters
- Black: Standard configuration
- Flake8: E203, W503 ignored (conflicts with Black)
- mypy: Strict mode enabled

### Code Quality Checklist

Before submitting a PR, ensure:

- ✅ Code formatted with Black: `uv run black src/`
- ✅ No linting errors: `uv run flake8 src/`
- ✅ Type hints added: `uv run mypy src/`
- ✅ All tests pass: `uv run pytest tests/ -v`
- ✅ Coverage ≥ 70%: `uv run pytest tests/ --cov=src`
- ✅ Docstrings added (Google style)
- ✅ No hardcoded values (use parameters.yml)

### Naming Conventions

**Files and Modules:**
```python
# Good
data_processing_nodes.py
train_bg_nbd_model.py

# Bad
dataProcessingNodes.py
TrainBGNBD.py
```

**Functions:**
```python
# Good: Verb phrases, snake_case
def train_bg_nbd_model(data: pl.DataFrame, params: Dict[str, Any]) -> BetaGeoFitter:
    """Train BG/NBD model for purchase frequency prediction."""
    ...

# Bad: Nouns, camelCase
def BGNBDModelTrainer(data, params):
    ...
```

**Variables:**
```python
# Good: Descriptive, snake_case
customer_lifetime_value = 450.0
churn_probability_30day = 0.15

# Bad: Abbreviations, unclear
clv = 450.0
p = 0.15
```

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 70% (enforced in CI/CD)
- **Target coverage**: 80% for critical modules (pipelines)
- **Unit tests**: Every public function must have tests
- **Integration tests**: Every pipeline must have end-to-end tests

### Writing Tests

**Test Structure:**
```python
# tests/test_data_processing.py

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
```

**Running Tests:**
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_data_processing.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run only unit tests
uv run pytest tests/ -m "unit" -v
```

### Test Fixtures

Use `conftest.py` for shared fixtures:

```python
# tests/conftest.py

import pytest
import polars as pl

@pytest.fixture
def sample_transactions():
    """Sample transaction data for testing."""
    return pl.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "transaction_dt": ["2024-01-01", "2024-02-01", "2024-01-15"],
        "amount_usd": [100.0, 150.0, 50.0]
    })
```

---

## Documentation Standards

### Docstring Format

We use **Google-style docstrings** for all functions, classes, and modules:

```python
def train_bg_nbd_model(data: pl.DataFrame, params: Dict[str, Any]) -> BetaGeoFitter:
    """Train BG/NBD model for purchase frequency prediction.

    BG/NBD is a probabilistic model for non-contractual settings that estimates
    purchase frequency and churn probability.

    Args:
        data: RFM summary DataFrame with columns:
            - frequency (int): Number of repeat purchases
            - recency (float): Time between first and last purchase
            - T (float): Customer age
        params: Configuration dictionary containing:
            - penalizer_coef (float): L2 regularization parameter (default: 0.0)

    Returns:
        Fitted BetaGeoFitter model with learned parameters.

    Raises:
        ValueError: If required columns (frequency, recency, T) are missing

    Example:
        >>> rfm_data = pl.DataFrame({...})
        >>> model = train_bg_nbd_model(rfm_data, {'penalizer_coef': 0.001})
        >>> predictions = model.predict(...)
    """
    ...
```

### Documentation Checklist

Before submitting:

- ✅ All new functions have docstrings
- ✅ Docstrings include Args, Returns, Raises
- ✅ Complex logic has inline comments
- ✅ Examples provided for non-obvious functions
- ✅ README updated if adding new features
- ✅ No TODO comments left in code

---

## Pull Request Process

### Before Submitting

1. **Update your branch** with latest develop:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Run the full test suite**:
   ```bash
   uv run pytest tests/ -v --cov=src
   ```

3. **Check code quality**:
   ```bash
   uv run black src/
   uv run flake8 src/
   uv run mypy src/
   ```

4. **Update documentation**:
   - Add docstrings to new functions
   - Update README.md if needed
   - Add entry to CHANGELOG.md (if exists)

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Coverage ≥ 70%

## Checklist
- [ ] Code follows style guidelines (Black, Flake8, mypy)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Dependent changes merged
```

### PR Review Process

1. **Automated checks** run on PR (GitHub Actions):
   - Tests must pass
   - Coverage must meet threshold
   - Linting must pass

2. **Code review** by maintainers:
   - At least 1 approval required
   - Address all review comments
   - Request re-review after changes

3. **Merge**:
   - Squash and merge to develop
   - Delete feature branch after merge

---

## Adding New Features

### Feature Development Checklist

When adding a new feature (e.g., new data source, new visualization):

1. **Plan** (use `/plan` in Claude Code or write design doc):
   - [ ] Define feature requirements
   - [ ] Identify affected pipelines
   - [ ] Plan configuration changes

2. **Implement**:
   - [ ] Create Kedro nodes in appropriate pipeline
   - [ ] Add datasets to `conf/base/catalog.yml`
   - [ ] Add parameters to `conf/base/parameters.yml`
   - [ ] Update pipeline definition

3. **Test**:
   - [ ] Write unit tests for new nodes
   - [ ] Write integration test for pipeline
   - [ ] Verify with sample data

4. **Document**:
   - [ ] Add docstrings to all functions
   - [ ] Update README.md with usage examples
   - [ ] Update `docs/technical_design.md` if architecture changes

5. **Submit PR** following process above

### Example: Adding New Data Source

```python
# 1. Add skeleton mapping in conf/base/parameters.yml
skeleton:
  new_source:
    mapping:
      customer_id: "user_id"
      transaction_date: "order_date"
      transaction_value: "total_amount"

# 2. Add dataset to conf/base/catalog.yml
raw_new_source:
  type: polars.CSVDataset
  filepath: data/01_raw/new_source.csv

# 3. Create node in src/ai_core/pipelines/data_processing/nodes.py
def create_new_source_skeleton(data: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """Convert new source data to transaction skeleton."""
    return map_to_transactions_skeleton(data, params["skeleton"]["new_source"])

# 4. Add to pipeline
node(
    func=create_new_source_skeleton,
    inputs=["raw_new_source", "params:skeleton"],
    outputs="new_source_skeleton",
    name="create_new_source_skeleton"
)
```

---

## Adding New Models

### Model Integration Checklist

When adding a new ML model (e.g., DeepSurv, Prophet):

1. **Research**:
   - [ ] Verify model applicability to CLTV
   - [ ] Check dependencies and licenses
   - [ ] Review computational requirements

2. **Implement Training Node**:
   ```python
   # src/ai_core/pipelines/data_science/nodes.py

   def train_deep_surv_model(
       survival_data: pl.DataFrame,
       params: Dict[str, Any]
   ) -> DeepSurv:
       """Train DeepSurv model for survival analysis.

       Args:
           survival_data: DataFrame with T, E, and covariates
           params: Model hyperparameters

       Returns:
           Trained DeepSurv model
       """
       # Implementation
       ...
   ```

3. **Add Prediction Node**:
   ```python
   def predict_with_deep_surv(
       model: DeepSurv,
       data: pl.DataFrame
   ) -> pl.DataFrame:
       """Generate predictions using DeepSurv model."""
       ...
   ```

4. **Update Configuration**:
   ```yaml
   # conf/base/parameters.yml
   modeling:
     model_type: "deep_surv"  # or keep "lifetimes"
     deep_surv:
       hidden_layers: [64, 32, 16]
       learning_rate: 0.001
       epochs: 100
   ```

5. **Test**:
   - [ ] Unit test for training node
   - [ ] Unit test for prediction node
   - [ ] Integration test for full pipeline
   - [ ] Validation against known dataset

6. **Benchmark**:
   - [ ] Compare performance vs existing models
   - [ ] Document training time and inference latency
   - [ ] Log metrics to MLflow

---

## Troubleshooting

### Common Issues

#### Issue: `uv sync` fails with dependency conflicts

**Solution:**
```bash
# Clear cache and reinstall
rm -rf .venv
uv cache clean
uv sync
```

#### Issue: Tests fail with `KedroContextError`

**Solution:**
Ensure `bootstrap_project()` called before `KedroSession`:
```python
from kedro.framework.startup import bootstrap_project
from pathlib import Path

bootstrap_project(Path.cwd())
```

#### Issue: Black and Flake8 disagree on formatting

**Solution:**
Black takes precedence. Update Flake8 config:
```ini
# setup.cfg or pyproject.toml
[flake8]
ignore = E203, W503
max-line-length = 100
```

#### Issue: mypy errors on third-party libraries

**Solution:**
Add to `pyproject.toml`:
```toml
[tool.mypy]
ignore_missing_imports = true
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing [GitHub Issues](https://github.com/Wizeline/ai_cores/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/Wizeline/ai_cores/discussions)
- **Slack**: Join `#ai-cores` channel (for Wizeline team members)

---

## Release Process

*(For maintainers only)*

1. **Version bump**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md` with release notes
3. **Tag**: Create git tag: `git tag -a v1.2.0 -m "Release v1.2.0"`
4. **Push**: `git push upstream main --tags`
5. **Release**: Create GitHub release from tag

---

## License

By contributing, you agree that your contributions will be licensed under the project's license (see `LICENSE` file).

---

## Acknowledgments

Thank you to all contributors who help improve this project! 🎉

**Key Contributors:**
- See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full list

**Inspired by:**
- [Kedro Contributing Guide](https://github.com/kedro-org/kedro/blob/main/CONTRIBUTING.md)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
