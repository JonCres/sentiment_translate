# Engineering & Documentation Standards

This document outlines the mandatory standards for all contributions to the AI Cores project.

## 1. Coding Standards

### General Principles
- **DRY (Don't Repeat Yourself)**: Abstract shared logic into `src/utils` or shared packages.
- **SOLID**: Follow object-oriented design principles.
- **Type Safety**: Mandatory static typing for all function signatures.

### Python Style
- **PEP 8**: All code must be PEP 8 compliant.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Imports**: Use absolute imports. Group by standard library, third-party, and local modules.

### Documentation (Docstrings)
- **Format**: Use **Google Style** docstrings.
- **Requirement**: All public functions, classes, and modules must have docstrings.

```python
def calculate_metric(data: pd.DataFrame, threshold: float) -> float:
    """Calculates a specific business metric.

    Args:
        data: The input DataFrame containing features.
        threshold: The cutoff value for the calculation.

    Returns:
        The calculated metric as a float.
    """
    # implementation
```

## 2. Pipeline Standards (Kedro)

### Data Catalog
- **No Hardcoding**: Never hardcode file paths in nodes. Use the `DataCatalog`.
- **Naming**: Use descriptive keys in `catalog.yml` (e.g., `churn_features_train`).

### Nodes
- **Purity**: Nodes should be pure functions (same input → same output).
- **Validation**: Use **Pandera** decorators to validate DataFrame schemas.

```python
import pandera as pa

@pa.check_types
def clean_data(df: pa.typing.DataFrame[InputSchema]) -> pa.typing.DataFrame[OutputSchema]:
    # implementation
```

## 3. Orchestration Standards (Prefect)

### The Pickle Rule
- **State Management**: Do not pass objects that cannot be serialized (pickled).
- **Session Init**: Initialize `KedroSession` inside the Prefect `@task`.

## 4. Documentation Standards

### Structure
Each AI Core must contain:
1. `docs/ai_product_canvas.md`: The business source of truth.
2. `docs/technical_design.md`: The technical source of truth.
3. `docs/runbook.md`: Operational instructions.
4. `root_context.md`: AI Agent instructions.

### Tone & Voice
- **Voice**: Active and direct ("The system processes..." instead of "The data is processed...").
- **Audience**: Technical yet accessible to cross-functional stakeholders.
- **Clarity**: Define acronyms on first use.

## 5. Global Invariants

1. **Context Awareness**: All changes must align with the `ai_product_canvas.md`.
2. **Reproducibility**: Always fix random seeds (`random_state=42`).
3. **Logging**: Use `logging.getLogger(__name__)`. Avoid `print()` statements in production code.
