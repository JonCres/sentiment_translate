from prefect import flow, deploy
from src.core.kedro_pipeline import KedroPipeline
from pathlib import Path
from src.utils.config_loader import load_config
from typing import Dict, Any

# Load config globally to allow dynamic flow naming
config = load_config("configs/project_config.yaml")
if config is None:
    raise ValueError("config must be provided")
aicore_name = config.get('project', {}).get('name', 'aicore_project_name').lower().replace(' ', '_')
flow_name = f"{aicore_name}_visualization_flow"

class VisualizationPipeline(KedroPipeline):
    """Visualization Pipeline: Generate charts and plots using Kedro"""
    pipeline_name = "visualization"

@flow(name=flow_name, log_prints=True)
def run_pipeline(config: Dict[str, Any]):
    """Entry point flow that instantiates and runs the VisualizationPipeline"""
    pipeline = VisualizationPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    deployment_config = config["deployments"]["visualization_pipeline"]
    work_pool_name = config["prefect"]["work_pool"]["name"]

    # Create deployment using from_source to handle storage/code loading
    pipeline_deployment = run_pipeline.from_source(
        source=str(Path.cwd()),
        entrypoint=deployment_config["entrypoint"]
    ).to_deployment(
        name=deployment_config["name"],
        description=deployment_config["description"],
        tags=deployment_config["tags"],
        parameters={"config": config, **deployment_config.get("parameters", {})}        
    )

    # Build & register all deployments at once
    deploy(
        pipeline_deployment,
        work_pool_name=work_pool_name,
    )