# AI Cores Glossary

This document defines the key terms, architectural patterns, and concepts used throughout the Wizeline AI Cores monorepo.

## Core Concepts

### AI Core
A standardized, production-ready, industry-specific capability layer. Each AI Core is an independent project located in `cores/[industry]/[name]` and follows a strictly defined structure for pipelines, orchestration, and documentation.

### Industry Solution
A collection of AI Cores tailored to a specific sector, such as **Media & Entertainment** or **Retail & CPG**.

### Monorepo
The architectural choice of housing multiple independent AI Cores within a single repository to share infrastructure, standards, and tools.

## Architectural Terms

### The Pickle Rule
A critical constraint for Prefect-Kedro integration. **Never** pass live Python objects (like `KedroContext` or database connections) between Prefect tasks. Instead, pass configuration keys or string references and re-initialize the session within the worker. This ensures serialization (pickling) safety across distributed environments.

### Gated Execution Modes
Operational states defined in a core's `root_context.md`:
- `[MODE: PLAN]`: Strategic alignment and architectural design.
- `[MODE: IMPLEMENT]`: Active development of nodes, tasks, or UI.
- `[MODE: DEBUG]`: Troubleshooting and performance tuning.

### Hardware Acceleration Flags
The `--extra` flags used with `uv sync` to enable platform-specific optimizations:
- `mps`: Apple Silicon (Metal Performance Shaders).
- `cuda`: NVIDIA GPUs.
- `xpu`: Intel Arc/Data Center GPUs.

## Technology Stack

### Kedro
The data engineering framework used to define modular, testable, and reproducible data pipelines. It acts as the **Transformation Layer**.

### Prefect
The workflow orchestration platform that manages scheduling, retries, and observability. It acts as the **Orchestration Layer**.

### MLflow
The lifecycle management platform used for experiment tracking, model versioning (Registry), and artifact storage.

### Feast
The Feature Store used to manage and serve machine learning features for both training and online inference.

### uv
The high-performance Python package manager used for all dependency resolution and environment management in this repository.
