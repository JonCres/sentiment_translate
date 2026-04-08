from prefect import flow, deploy, get_run_logger
from src.core.kedro_pipeline import KedroPipeline
from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.mlflow_tracking import setup_mlflow_tracking, link_prefect_run_id
from typing import Dict, Any
import mlflow

# Load config globally to allow dynamic flow naming
config = load_config("configs/project_config.yaml")
if config is None:
    raise ValueError("config must be provided")
aicore_name = config.get('project', {}).get('name', 'aicore_project_name').lower().replace(' ', '_')
flow_name = f"{aicore_name}_monitoring_flow"


class MonitoringPipeline(KedroPipeline):
    """Monitoring Pipeline: Drift Detection, Performance Checks using Kedro with MLflow tracking."""
    pipeline_name = "monitoring"

    def setup(self):
        """Setup pipeline resources including MLflow."""
        super().setup()
        mlops_config = self.config.get('kedro_params', {}).get('mlops', {})
        setup_mlflow_tracking(mlops_config)

    def run(self):
        """Execute pipeline with MLflow run context."""
        self.setup()

        mlops_config = self.config.get('kedro_params', {}).get('mlops', {})
        experiment_name = mlops_config.get('experiment_name', 'predictive_cltv_monitoring')

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{self.aicore_name}_monitoring"):
            link_prefect_run_id()
            super().run()

            if self.logger:
                self.logger.info(f"MLflow run completed: {mlflow.active_run().info.run_id}")


@flow(name=flow_name, log_prints=True)
def run_pipeline(config: Dict[str, Any]):
    """Entry point flow that instantiates and runs the MonitoringPipeline with MLflow tracking"""
    pipeline = MonitoringPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    deployment_config = config["deployments"]["monitoring_pipeline"]
    work_pool_name = config["prefect"]["work_pool"]["name"]

    pipeline_deployment = run_pipeline.from_source(
        source=str(Path.cwd()),
        entrypoint=deployment_config["entrypoint"],
    ).to_deployment(
        name=deployment_config["name"],
        description=deployment_config["description"],
        tags=deployment_config["tags"],
        parameters={"config": config, **deployment_config.get("parameters", {})}
    )

    deploy(
        pipeline_deployment,
        work_pool_name=work_pool_name,
    )

