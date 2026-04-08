from pathlib import Path
from typing import Dict, Any
from prefect import flow, deploy, task
from src.prefect_orchestration.data_pipeline import DataPipeline
from src.prefect_orchestration.ai_pipeline import AIPipeline
from src.prefect_orchestration.visualization_pipeline import VisualizationPipeline
from src.prefect_orchestration.monitoring_pipeline import MonitoringPipeline
from src.utils.config_loader import load_config

# Load config globally to allow dynamic flow naming
config = load_config("configs/project_config.yaml")
if config is None:
    raise ValueError("config must be provided")
ai_core_name = config.get('project', {}).get('name', 'ai_core_project_name').lower().replace(' ', '_')
flow_name = f"{ai_core_name}_complete_flow"

@task(name="run_visualization_task")
def run_visualization_wrapper(config: Dict[str, Any]):
    # We instantiate and run the pipeline. 
    # Since pipeline.run() is a flow, this task will create a subflow run.
    # Prefect allows calling flows from tasks.
    pipeline = VisualizationPipeline(config)
    pipeline.run()

@task(name="run_monitoring_task")
def run_monitoring_wrapper(config: Dict[str, Any]):
    pipeline = MonitoringPipeline(config)
    pipeline.run()

@flow(name=flow_name, log_prints=True)
def run_complete_pipeline(config: Dict[str, Any]):
    """
    Master pipeline that orchestrates the execution of:
    1. Data Pipeline
    2. AI Pipeline
    3. Visualization Pipeline & Monitoring Pipeline (in parallel)
    """
    # 1. Run Data Pipeline
    data_pipeline = DataPipeline(config)
    data_pipeline.run()

    # 2. Run AI Pipeline (after Data Pipeline completes)
    ai_pipeline = AIPipeline(config)
    ai_pipeline.run()

    # 3. Run Visualization and Monitoring Pipelines in parallel
    # We use .submit() to run tasks in parallel
    viz_future = run_visualization_wrapper.submit(config)
    mon_future = run_monitoring_wrapper.submit(config)
    
    # Wait for them to complete
    viz_future.wait()
    mon_future.wait()

if __name__ == "__main__":
    
    deployment_config = config["deployments"]["master_pipeline"]
    work_pool_name = config["prefect"]["work_pool"]["name"]
    
    pipeline_deployment = run_complete_pipeline.from_source(
        source=str(Path.cwd()),
        entrypoint=deployment_config["entrypoint"],
    ).to_deployment(
        name=deployment_config["name"],
        description=deployment_config["description"],
        tags=deployment_config["tags"],
        parameters={"config": config, **deployment_config.get("parameters", {})},
    )
    
    deploy(
        pipeline_deployment,
        work_pool_name=work_pool_name,
    )
