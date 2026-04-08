from kedro.framework.hooks import hook_impl
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from typing import Any, Dict, Set
import logging

logger = logging.getLogger(__name__)


class PipelineNameHook:
    """Hook to log execution status of top-level and modular pipelines.

    This hook detects transitions between modular pipelines (e.g., data_processing,
    data_science) by inspecting the module paths of nodes as they execute.

    Stage completion is only logged once ALL nodes belonging to that stage have
    finished, which avoids misleading log ordering when Kedro's DAG resolution
    interleaves nodes from different modular pipelines.
    """

    def __init__(self):
        self._stage_total: Dict[str, int] = {}
        self._stage_completed: Dict[str, int] = {}
        self._stage_started: Set[str] = set()

    @staticmethod
    def _get_stage(node: Node) -> str | None:
        """Infer modular pipeline name from node module path.

        Example: ai_core.pipelines.data_processing.nodes -> data_processing
        """
        module_name = node.func.__module__
        parts = module_name.split(".")
        if "pipelines" in parts:
            idx = parts.index("pipelines")
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return None

    @hook_impl(tryfirst=True)
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline
    ) -> None:
        """Pre-compute per-stage node counts and log top-level pipeline start."""
        if "pipeline_name" not in run_params or run_params["pipeline_name"] is None:
            run_params["pipeline_name"] = "__default__"

        # Reset tracking state
        self._stage_total = {}
        self._stage_completed = {}
        self._stage_started = set()

        for node in pipeline.nodes:
            stage = self._get_stage(node)
            if stage:
                self._stage_total[stage] = self._stage_total.get(stage, 0) + 1
                self._stage_completed[stage] = 0

        pipeline_name = run_params.get("pipeline_name", "unknown")
        logger.info(f"\n{'#' * 60}")
        logger.info(f"🚀 STARTING GLOBAL RUN: {pipeline_name.upper()}")
        logger.info(f"{'#' * 60}\n")

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        """Log when a modular pipeline stage is entered for the first time."""
        stage = self._get_stage(node)
        if stage and stage not in self._stage_started:
            self._stage_started.add(stage)
            logger.info(f"\n{'=' * 40}")
            logger.info(f"🏁 ENTERING STAGE: {stage.upper()}")
            logger.info(f"{'=' * 40}\n")

    @hook_impl
    def after_node_run(self, node: Node) -> None:
        """Log stage completion only when ALL nodes in the stage have finished."""
        stage = self._get_stage(node)
        if stage:
            self._stage_completed[stage] = self._stage_completed.get(stage, 0) + 1
            if self._stage_completed[stage] == self._stage_total.get(stage, 0):
                logger.info(f"\n{'=' * 40}")
                logger.info(f"✅ COMPLETED: {stage.upper()}")
                logger.info(f"{'=' * 40}\n")

    @hook_impl
    def after_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Log top-level pipeline finish."""
        pipeline_name = run_params.get("pipeline_name", "unknown")
        logger.info(f"\n{'#' * 60}")
        logger.info(f"🏆 GLOBAL RUN FINISHED: {pipeline_name.upper()}")
        logger.info(f"{'#' * 60}\n")
