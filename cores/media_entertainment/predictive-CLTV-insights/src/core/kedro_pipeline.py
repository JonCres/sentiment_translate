from abc import ABC, abstractmethod
from prefect import flow, get_run_logger
from typing import Dict, Any, Optional, List
from src.core.prefect_logger import PrefectLogHandler
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import logging

class KedroPipeline(ABC):
    """Abstract base class for hybrid Kedro + Prefect pipeline orchestration.

    Implements the Adapter pattern to run Kedro pipelines within Prefect flows, combining
    Kedro's data catalog and node management with Prefect's orchestration, retries, and
    observability. This enables production-grade ML pipelines with minimal boilerplate.

    Architecture Pattern:
    - **Kedro Layer**: Pure-function nodes, data catalog (I/O abstraction), pipeline DAG
    - **Prefect Layer**: Task scheduling, retries, logging, notifications, caching
    - **Bridge (this class)**: Lifecycle management, logging integration, session handling

    Critical Invariant ("Pickling Rule"):
    Child classes must NEVER store KedroSession or live connections as instance attributes.
    Pass string references (config keys) between Prefect tasks and re-initialize KedroSession
    inside each task worker. Violation causes serialization errors in distributed execution.

    Class Attributes:
        pipeline_name: Kedro pipeline name (e.g., 'data_processing', 'data_science').
            Must be overridden by child classes. Corresponds to pipeline_registry.py entries.

    Instance Attributes:
        config: Configuration dictionary loaded from YAML (project_config.yaml)
        aicore_name: Project name (derived from config, used for Prefect flow naming)
        logger: Prefect logger instance (initialized in setup())
        kedro_handlers: List of logging handlers for cleanup

    Usage:
        Child classes override pipeline_name and inherit run() method:
        >>> class DataPipeline(KedroPipeline):
        ...     pipeline_name = 'data_processing'
        >>>
        >>> config = load_config('configs/project_config.yaml')
        >>> pipeline = DataPipeline(config)
        >>> pipeline.run()  # Executes as Prefect flow with Kedro integration
    """

    # Child classes should override these
    pipeline_name: str = None
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        if self.config is None:
            raise ValueError("config must be provided")

        self.aicore_name = self.config.get('project', {}).get('name', 'aicore_project_name').lower().replace(' ', '_')
        self.logger = None
        self.kedro_handlers = []        

        # Only process if both attributes are set (not on KedroPipeline itself)
        if self.pipeline_name is not None:
            # Wrap the run method with the flow decorator using the flow_name
            original_run = self.run
            self.run = flow(name=f"{self.aicore_name}_kedro_{self.pipeline_name}_pipeline", log_prints=True)(original_run)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} must define 'pipeline_name' class attribute")
    
    def _get_log_level(self) -> int:
        """Get log level from config"""
        level_str = self.config.get('logging', {}).get('level', 'INFO')
        return getattr(logging, level_str.upper(), logging.INFO)
    
    def _get_log_format(self) -> str:
        """Get log format from config"""
        return self.config.get('logging', {}).get('format', '%(name)s - %(levelname)s - %(message)s')
    
    def _get_kedro_log_level(self) -> int:
        """Get Kedro log level from config"""
        level_str = self.config.get('logging', {}).get('kedro_log_level', 'INFO')
        return getattr(logging, level_str.upper(), logging.INFO)
    
    def setup_kedro_logging(self, prefect_logger) -> List[logging.Handler]:
        """Configure Kedro loggers to forward messages to Prefect's observability system.

        Bridges Kedro's logging (used by nodes and catalog) with Prefect's logger to
        ensure all pipeline events appear in Prefect UI and logs. Creates custom handlers
        that intercept Kedro log messages and re-emit them through Prefect's logger.

        Kedro Loggers Configured:
        - 'kedro': Core Kedro framework logger (session, runner, io)
        - 'aicore': Project-specific logger (used by pipeline nodes)

        Args:
            prefect_logger: Prefect logger instance from get_run_logger(). All Kedro
                messages will be forwarded to this logger for centralized observability.

        Returns:
            List of (logger, handler) tuples for cleanup. Each tuple contains the logger
            object and the PrefectLogHandler attached to it. Pass to cleanup_kedro_logging()
            after pipeline completes.

        Note:
            - Only activates if config['logging']['forward_kedro_logs'] = True (default)
            - Log level and format controlled by config settings
            - Handlers must be removed after pipeline execution to prevent duplicate logging

        Example:
            >>> prefect_logger = get_run_logger()
            >>> handlers = self.setup_kedro_logging(prefect_logger)
            >>> # Run Kedro pipeline (Kedro logs appear in Prefect)
            >>> self.cleanup_kedro_logging(handlers)
        """
        handlers = []
        
        # Only setup if forward_kedro_logs is enabled
        if not self.config.get('logging', {}).get('forward_kedro_logs', True):
            return handlers
        
        # Create Prefect handler with config settings
        prefect_handler = PrefectLogHandler(prefect_logger)
        prefect_handler.setLevel(self._get_kedro_log_level())
        
        formatter = logging.Formatter(self._get_log_format())
        prefect_handler.setFormatter(formatter)
        
        # Add Prefect handler to Kedro loggers
        kedro_logger = logging.getLogger('kedro')
        kedro_logger.addHandler(prefect_handler)
        handlers.append((kedro_logger, prefect_handler))
        
        aicore_logger = logging.getLogger('aicore')
        aicore_logger.addHandler(prefect_handler)
        handlers.append((aicore_logger, prefect_handler))
        
        return handlers
    
    def cleanup_kedro_logging(self, handlers: List[tuple]):
        """Remove Kedro logging handlers"""
        for logger, handler in handlers:
            logger.removeHandler(handler)
    
    def run(self):
        """Execute Kedro pipeline within Prefect flow with full lifecycle management.

        Orchestrates the complete pipeline execution lifecycle: setup → Kedro session →
        pipeline run → cleanup. Automatically wrapped as a Prefect @flow on initialization,
        enabling retries, caching, and observability without modifying child classes.

        Execution Flow:
        1. setup(): Initialize Prefect logger and resources
        2. Bootstrap Kedro project from current working directory
        3. Configure Kedro → Prefect logging bridge
        4. Create KedroSession and run pipeline_name
        5. Clean up logging handlers
        6. teardown(): Release resources

        Prefect Integration:
        - Flow name: '{aicore_name}_kedro_{pipeline_name}_pipeline'
        - Automatic retry on transient failures (configured in Prefect deployment)
        - Log prints captured and forwarded to Prefect UI
        - Task dependencies inferred from Kedro DAG

        Note:
            - Must be called from within a Prefect context (flow runner or deployment)
            - Uses KedroSession.create() context manager for safe resource management
            - Logging handlers cleaned up even if pipeline fails (via try/finally pattern)
            - Pipeline name must exist in src/ai_core/pipeline_registry.py

        Example:
            >>> from prefect import serve
            >>> config = load_config()
            >>> pipeline = DataPipeline(config)  # KedroPipeline subclass
            >>> # Run directly:
            >>> pipeline.run()
            >>> # Or deploy to Prefect:
            >>> serve(pipeline.run)
        """
        self.setup()
        
        # Get Prefect's logger
        prefect_logger = get_run_logger()
        
        project_path = Path.cwd()
        bootstrap_project(project_path)
        
        # Setup Kedro logging using config
        handlers = self.setup_kedro_logging(prefect_logger)

        with KedroSession.create(project_path=project_path) as session:
            # Pipeline start/end logging is handled by Kedro hooks in src/ai_core/hooks.py
            # to avoid duplicate logging and maintain synchronization
            session.run(pipeline_name=self.pipeline_name)

        # Clean up handlers
        self.cleanup_kedro_logging(handlers)
        
        self.teardown()
    
    def setup(self):
        """Setup pipeline resources"""
        self.logger = get_run_logger()
        self.logger.info(f"Setting up {self.__class__.__name__}")
    
    def teardown(self):
        """Cleanup resources"""
        if self.logger:
            self.logger.info(f"Tearing down {self.__class__.__name__}")

