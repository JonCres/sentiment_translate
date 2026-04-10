"""Tests for PipelineNameHook in ai_core.hooks."""

import logging
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from ai_core.hooks import PipelineNameHook


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_mock_node(module_path: str) -> MagicMock:
    """Create a mock Kedro Node whose func.__module__ returns *module_path*."""
    node = MagicMock()
    node.func = MagicMock()
    node.func.__module__ = module_path
    return node


def _make_pipeline(nodes: list[MagicMock]) -> MagicMock:
    """Create a mock Pipeline with the given list of nodes."""
    pipeline = MagicMock()
    pipeline.nodes = nodes
    return pipeline


# ---------------------------------------------------------------------------
# Tests: _get_stage
# ---------------------------------------------------------------------------

class TestGetStage:
    """Tests for PipelineNameHook._get_stage."""

    def test_extracts_data_processing(self):
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")
        assert PipelineNameHook._get_stage(node) == "data_processing"

    def test_extracts_data_science(self):
        node = _make_mock_node("ai_core.pipelines.data_science.unified_insights.nodes")
        assert PipelineNameHook._get_stage(node) == "data_science"

    def test_extracts_monitoring(self):
        node = _make_mock_node("ai_core.pipelines.monitoring.nodes")
        assert PipelineNameHook._get_stage(node) == "monitoring"

    def test_returns_none_for_non_pipeline_module(self):
        node = _make_mock_node("ai_core.utils.helpers")
        assert PipelineNameHook._get_stage(node) is None

    def test_returns_none_when_pipelines_is_last_part(self):
        """Edge case: 'pipelines' appears at the end with no stage after it."""
        node = _make_mock_node("ai_core.pipelines")
        assert PipelineNameHook._get_stage(node) is None


# ---------------------------------------------------------------------------
# Tests: before_pipeline_run
# ---------------------------------------------------------------------------

class TestBeforePipelineRun:
    """Tests for PipelineNameHook.before_pipeline_run."""

    def test_initializes_stage_totals(self):
        hook = PipelineNameHook()
        nodes = [
            _make_mock_node("ai_core.pipelines.data_processing.nodes"),
            _make_mock_node("ai_core.pipelines.data_processing.nodes"),
            _make_mock_node("ai_core.pipelines.data_science.nodes"),
        ]
        pipeline = _make_pipeline(nodes)
        run_params = {"pipeline_name": "my_pipeline"}

        hook.before_pipeline_run(run_params=run_params, pipeline=pipeline)

        assert hook._stage_total == {"data_processing": 2, "data_science": 1}
        assert hook._stage_completed == {"data_processing": 0, "data_science": 0}
        assert hook._stage_started == set()

    def test_sets_default_pipeline_name_when_missing(self):
        hook = PipelineNameHook()
        run_params: dict = {}
        pipeline = _make_pipeline([])

        hook.before_pipeline_run(run_params=run_params, pipeline=pipeline)
        assert run_params["pipeline_name"] == "__default__"

    def test_sets_default_pipeline_name_when_none(self):
        hook = PipelineNameHook()
        run_params = {"pipeline_name": None}
        pipeline = _make_pipeline([])

        hook.before_pipeline_run(run_params=run_params, pipeline=pipeline)
        assert run_params["pipeline_name"] == "__default__"

    def test_resets_state_on_new_run(self):
        """Calling before_pipeline_run again resets all tracking state."""
        hook = PipelineNameHook()
        nodes = [_make_mock_node("ai_core.pipelines.data_processing.nodes")]
        pipeline = _make_pipeline(nodes)

        hook.before_pipeline_run(
            run_params={"pipeline_name": "first"}, pipeline=pipeline
        )
        # Simulate some progress
        hook._stage_started.add("data_processing")
        hook._stage_completed["data_processing"] = 1

        # Second run should reset
        hook.before_pipeline_run(
            run_params={"pipeline_name": "second"}, pipeline=pipeline
        )
        assert hook._stage_started == set()
        assert hook._stage_completed == {"data_processing": 0}

    def test_logs_pipeline_start(self, caplog):
        hook = PipelineNameHook()
        pipeline = _make_pipeline([])
        with caplog.at_level(logging.INFO):
            hook.before_pipeline_run(
                run_params={"pipeline_name": "test_pipe"}, pipeline=pipeline
            )
        assert "STARTING GLOBAL RUN" in caplog.text
        assert "TEST_PIPE" in caplog.text


# ---------------------------------------------------------------------------
# Tests: before_node_run
# ---------------------------------------------------------------------------

class TestBeforeNodeRun:
    """Tests for PipelineNameHook.before_node_run."""

    def test_logs_stage_entry_first_time(self, caplog):
        hook = PipelineNameHook()
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")

        with caplog.at_level(logging.INFO):
            hook.before_node_run(node=node)

        assert "ENTERING STAGE" in caplog.text
        assert "DATA_PROCESSING" in caplog.text

    def test_does_not_log_stage_entry_twice(self, caplog):
        hook = PipelineNameHook()
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")

        hook.before_node_run(node=node)
        caplog.clear()

        with caplog.at_level(logging.INFO):
            hook.before_node_run(node=node)

        assert "ENTERING STAGE" not in caplog.text

    def test_logs_different_stages_independently(self, caplog):
        hook = PipelineNameHook()
        node_dp = _make_mock_node("ai_core.pipelines.data_processing.nodes")
        node_ds = _make_mock_node("ai_core.pipelines.data_science.nodes")

        with caplog.at_level(logging.INFO):
            hook.before_node_run(node=node_dp)
            hook.before_node_run(node=node_ds)

        assert "DATA_PROCESSING" in caplog.text
        assert "DATA_SCIENCE" in caplog.text

    def test_no_log_for_non_pipeline_node(self, caplog):
        hook = PipelineNameHook()
        node = _make_mock_node("ai_core.utils.helpers")

        with caplog.at_level(logging.INFO):
            hook.before_node_run(node=node)

        assert "ENTERING STAGE" not in caplog.text


# ---------------------------------------------------------------------------
# Tests: after_node_run
# ---------------------------------------------------------------------------

class TestAfterNodeRun:
    """Tests for PipelineNameHook.after_node_run."""

    def test_increments_completed_counter(self):
        hook = PipelineNameHook()
        hook._stage_total = {"data_processing": 3}
        hook._stage_completed = {"data_processing": 0}
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")

        hook.after_node_run(node=node)

        assert hook._stage_completed["data_processing"] == 1

    def test_logs_completion_when_all_nodes_done(self, caplog):
        hook = PipelineNameHook()
        hook._stage_total = {"data_processing": 2}
        hook._stage_completed = {"data_processing": 1}
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")

        with caplog.at_level(logging.INFO):
            hook.after_node_run(node=node)

        assert "COMPLETED" in caplog.text
        assert "DATA_PROCESSING" in caplog.text
        assert hook._stage_completed["data_processing"] == 2

    def test_no_completion_log_when_nodes_remaining(self, caplog):
        hook = PipelineNameHook()
        hook._stage_total = {"data_processing": 3}
        hook._stage_completed = {"data_processing": 0}
        node = _make_mock_node("ai_core.pipelines.data_processing.nodes")

        with caplog.at_level(logging.INFO):
            hook.after_node_run(node=node)

        assert "COMPLETED" not in caplog.text

    def test_no_crash_for_non_pipeline_node(self):
        """Nodes not belonging to a pipeline stage should not raise."""
        hook = PipelineNameHook()
        node = _make_mock_node("ai_core.utils.helpers")
        hook.after_node_run(node=node)  # should not raise


# ---------------------------------------------------------------------------
# Tests: after_pipeline_run
# ---------------------------------------------------------------------------

class TestAfterPipelineRun:
    """Tests for PipelineNameHook.after_pipeline_run."""

    def test_logs_pipeline_finish(self, caplog):
        hook = PipelineNameHook()
        with caplog.at_level(logging.INFO):
            hook.after_pipeline_run(run_params={"pipeline_name": "test_pipe"})
        assert "GLOBAL RUN FINISHED" in caplog.text
        assert "TEST_PIPE" in caplog.text

    def test_logs_unknown_when_name_missing(self, caplog):
        hook = PipelineNameHook()
        with caplog.at_level(logging.INFO):
            hook.after_pipeline_run(run_params={})
        assert "UNKNOWN" in caplog.text
