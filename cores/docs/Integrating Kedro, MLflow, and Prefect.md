## **Technical Integration of Kedro and MLflow**

The synergy between Kedro and MLflow is facilitated primarily through the kedro-mlflow plugin, which bridges the gap between Kedro's pipeline execution and MLflow's tracking capabilities.25 This plugin is designed to enforce Kedro principles while making MLflow usage as production-ready as possible.2

### **Configuration and Initial Setup**

The integration process begins with the installation of the plugin: pip install kedro-mlflow.1 In modern versions of Kedro (0.16.4+), the plugin's hooks are registered automatically via auto-discovery, eliminating the need for manual registration in settings.py unless custom behavior is required.14 Upon installation, the user initializes the project configuration by running kedro mlflow init.1

This command generates a mlflow.yml configuration file, typically located in the conf/local directory.1 This file acts as the central control point for how Kedro interacts with MLflow. It contains several critical sections that define tracking behavior, experiment naming, and UI settings.27

| MLflow.yml Key | Description | Impact |
| :---- | :---- | :---- |
| mlflow\_tracking\_uri | The URI of the tracking server.28 | Directs data to local storage or a remote server.29 |
| experiment.name | The name of the experiment.28 | Groups all related pipeline runs in the UI.1 |
| run.name | Template for the run name.28 | Can be static or dynamic using the ${km.random\_name:} resolver.28 |
| run.nested | Boolean for sub-runs.28 | Enables hierarchical run structures for hyperparameter tuning.27 |
| disable\_autologging | Deactivates standard MLflow autologging.31 | Essential for Databricks to avoid conflicts with the plugin.31 |

### **Experiment Tracking and Automatic Logging**

One of the primary advantages of using kedro-mlflow is the automatic tracking of pipeline parameters. By default, every parameter defined in Kedro's parameters.yml (or environment-specific overrides) is automatically logged to MLflow as a parameter at the start of the run.14 This ensures perfect alignment between the configuration used and the results obtained, which is a prerequisite for reproducible machine learning.14

The plugin also handles the lifecycle of the MLflow run. When a kedro run command is issued, the MlflowHook starts a new MLflow run, executes the pipeline, and closes the run upon completion.28 This one-to-one relationship between a Kedro run and an MLflow run simplifies the tracking experience. If a Kedro pipeline is executed within an environment where an MLflow run is already active (such as a Prefect flow or an Airflow task), the plugin intelligently detects this and logs data to the existing run instead of creating a new one.28

### **Specialized Datasets for Artifact Management**

To track data snapshots, diagnostic plots, and models as MLflow artifacts, the plugin introduces specialized dataset types that extend Kedro's I/O capabilities.1 These datasets are defined in the catalog.yml and provide a seamless way to store data both locally and in the MLflow artifact store.1

The MlflowArtifactDataset is a versatile wrapper that can be applied to any standard Kedro dataset.1 It ensures that the output of a node is saved to its primary filepath and simultaneously uploaded to MLflow.1 This is particularly useful for logging reporting outputs, such as confusion matrices generated via matplotlib.MatplotlibWriter or intermediate data summaries.14

For machine learning models, the MlflowModelTrackingDataset is the standard choice.14 This dataset is "flavor-aware," meaning it utilizes MLflow's built-in support for various libraries to serialize models correctly.18

YAML

\# Example: Configuration for an MLflow Model in catalog.yml  
regressor:  
  type: kedro\_mlflow.io.models.MlflowModelTrackingDataset  
  flavor: mlflow.sklearn  
  save\_args:  
    registered\_model\_name: "spaceflights-regressor"

In this configuration, when the regressor dataset is saved, the model is logged as an artifact in the current run and automatically registered in the Model Registry under the specified name.16 If the user needs to load a specific model version for inference, they can provide a run\_id or utilize runtime parameters to specify the target version dynamically.14

### **Metrics and History Tracking**

Quantitative performance indicators are managed through the MlflowMetricDataset and MlflowMetricHistoryDataset.35 While the former logs a single scalar value at the end of a run, the latter is designed for iterative processes, such as deep learning training loops, where a metric (like loss or accuracy) needs to be recorded over multiple epochs.31 This enables the generation of learning curves directly within the MLflow UI, providing immediate insight into the model's convergence behavior.4

## **The "Pipeline as Model" Paradigm**

A sophisticated feature of the kedro-mlflow integration is the ability to treat an entire Kedro inference pipeline as a single MLflow Model object.18 In production environments, a model rarely functions in isolation; it usually requires a sequence of preprocessing steps (e.g., imputation, scaling, feature encoding) and postprocessing steps (e.g., thresholding, label decoding).18

### **The pipeline\_ml\_factory Utility**

The pipeline\_ml\_factory function is used to bind a "training" pipeline and an "inference" pipeline together.38 This binding ensures that the inference pipeline is consistent with the training pipeline and that all necessary artifacts produced during training are included in the packaged model.18

When the training pipeline runs, the kedro-mlflow hook automatically captures the inference nodes and the artifacts they depend on (which must be persisted in the Data Catalog).18 The result is a self-contained MLflow model that includes the entire computational graph required to generate a prediction from raw input.18

Requirements for serving a pipeline as a model include:

1. One input must be a format supported by MLflow serving, such as a pandas.DataFrame, spark.DataFrame, or numpy.array.18  
2. All inputs to the inference nodes (except the raw data to be predicted) must be persisted datasets in the catalog.18  
3. Parameters used by the inference pipeline are persisted at export time and cannot be modified at runtime once the model is packaged.18

This approach drastically reduces the operational risk of "training-serving skew," where the preprocessing logic in the production API differs from the logic used during training.38 By sharing nodes between the two pipelines and packaging them together, the architecture guarantees functional parity.17

## **Visibility and Log Synchronization**

A common challenge when running Kedro within an orchestrator like Prefect is the fragmentation of logs.50 By default, Kedro logs to its own internal handlers, and these logs might not be automatically captured by the Prefect UI.50

### **Redirecting Kedro Logs to Prefect**

To ensure that every transformation step and dataset operation is visible in the Prefect dashboard, developers must configure Kedro's logging.yml to use the Prefect log handler.50 This ensures that any logging.info or logging.error calls made within Kedro nodes are correctly propagated to the Prefect backend.50

| Log Source | Default Destination | Target Destination |
| :---- | :---- | :---- |
| Kedro Node Logic | Standard Out / File | Prefect UI via Prefect Handler.50 |
| Dataset I/O | Standard Out | Prefect UI via Prefect Handler.50 |
| Prefect Task Metadata | Prefect UI | Prefect UI (Native).6 |
| MLflow Run Data | MLflow UI | MLflow UI (Native).4 |

### **Linking Run IDs for Full Traceability**

For end-to-end auditability, it is critical to link the Prefect flow\_run\_id with the MLflow run\_id.14 This can be accomplished by retrieving the Prefect context and logging it as a tag in the MLflow run.14

Python

import prefect  
import mlflow

\# Within a Prefect task  
run\_context \= prefect.context.get\_run\_context()  
prefect\_id \= run\_context.flow\_run.id  
mlflow.set\_tag("prefect\_flow\_run\_id", prefect\_id)

Conversely, the MLflow run\_id can be logged as an artifact or a tag in the Prefect flow.51 This bi-directional linking allows an engineer to navigate from a failed task in Prefect directly to the corresponding experiment results in MLflow.4

## **Security, Authentication, and Secret Management**

Productionizing this stack requires a robust strategy for handling credentials and securing communication between components.52

### **Managing Secrets in Kedro and Prefect**

Kedro traditionally uses a credentials.yml file for sensitive information like S3 access keys or database passwords.12 In a production environment managed by Prefect, these secrets should not reside in the code repository. Instead, they should be stored as Prefect Secret Blocks.52

At runtime, the Prefect task can retrieve these secrets and pass them to the Kedro session.52 This can be done by using Kedro's TemplatedConfigLoader, which allows for the dynamic injection of environment variables or dictionary values into the project's configuration.12 Alternatively, a custom Kedro Hook (after\_context\_created) can be implemented to fetch secrets from the Prefect API and populate the Kedro ConfigLoader programmatically.15

### **Authenticating with Remote MLflow Servers**

When deploying MLflow as a centralized service, basic HTTP authentication or OAuth2.0 is often enabled.28 The kedro-mlflow plugin supports these security measures through the server section of the mlflow.yml file.28

Credentials for MLflow can be specified in Kedro's credentials.yml, and the plugin will automatically export them as environment variables (e.g., MLFLOW\_TRACKING\_USERNAME, MLFLOW\_TRACKING\_PASSWORD) during execution.28 For environments with expiring tokens, the plugin allows the configuration of a CustomRequestHeaderProvider, ensuring that the correct authorization headers are included in every API request sent to the tracking server.27

## **Advanced Features in MLflow 3.0 and kedro-mlflow 2.0.0**

The release of MLflow 3.x and the corresponding kedro-mlflow 2.0.0 update introduced significant shifts in the MLOps landscape, particularly in how models are referenced and artifacts are managed.31

### **Model-Centric Referencing**

MLflow 3.0 introduces a new model URI format that uses unique Model IDs instead of Run IDs.58 This provides more direct referencing and decouples the model's identity from the specific training execution that produced it.58 In kedro-mlflow 2.0.0, the run\_id argument has been dropped from the MlflowModelTrackingDataset in favor of this new model-centric approach.31 Users can now specify a model\_uri directly in the load\_args to retrieve specific models from the registry without needing to track the original run identifier.31

### **Linking Metrics to Checkpoints and Datasets**

A major enhancement in MLflow 3 is the ability to link specific performance metrics to individual model checkpoints and dataset versions.58 This provides an unprecedented level of traceability.58 For example, in a Kedro pipeline, one can log the training loss for every epoch and link each record to the specific version of the training dataset used, as well as the unique ID of the model weights saved at that step.58 This level of detail is invaluable for auditing and debugging the performance of models over time.34

## **Implementing the Unified Stack: A Strategic Roadmap**

To successfully integrate MLflow, Kedro, and Prefect, organizations should follow a phased implementation strategy that prioritizes structural integrity before introducing orchestration complexity.

### **Phase 1: Structural Governance**

The foundation must be a clean, modular code base.1 Teams should begin by refactoring their existing data science code into Kedro nodes and pipelines.1 The primary goal of this phase is to move all data references into the Data Catalog and all hyperparameters into parameters.yml.8 This ensures that the code is portable and testable.1

### **Phase 2: Reproducibility and Tracking**

Once the project is structured, the next step is to introduce MLflow for experiment tracking.4 Install kedro-mlflow and configure it to point to a centralized tracking server.1 Replace local model saving logic with the MlflowModelTrackingDataset to begin building a version-controlled model registry.14 In this phase, teams should also implement the pipeline\_ml\_factory to standardize how training and inference pipelines are coupled.18

### **Phase 3: Reliable Orchestration**

The final phase involves moving from manual kedro run commands to orchestrated Prefect flows.7 Start by creating simple Prefect flows that execute Kedro pipelines via the KedroSession.41 Implement Prefect Secret Blocks to manage credentials and transition to a containerized deployment model using Prefect Workers on Kubernetes or a similar platform.23

### **Phase 4: Full Lifecycle Automation (CI/CD/CT)**

With the three tools in place, organizations can automate the entire lifecycle. Use Prefect Automations to trigger model retraining whenever new data is detected or when model performance drops below a certain threshold (as monitored by MLflow).4 Integrate this with CI/CD tools to ensure that every change to the Kedro code is automatically tested and the resulting model is registered and evaluated before deployment.8