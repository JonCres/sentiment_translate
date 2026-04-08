from abc import ABC, abstractmethod
from prefect import flow, get_run_logger
from typing import Dict, Any, Optional, List
from src.core.prefect_logger import PrefectLogHandler
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import logging

class KedroPipeline(ABC):
    """Abstract base class for all AI Core pipelines"""
    
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
        """
        Setup Kedro logging to forward to Prefect logger
        Returns list of handlers for cleanup
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
        """Execute pipeline flow via Kedro - common implementation for all pipelines"""
        self.setup()
        
        # Get Prefect's logger
        prefect_logger = get_run_logger()
        
        project_path = Path.cwd()
        bootstrap_project(project_path)
        
        # Setup Kedro logging using config
        handlers = self.setup_kedro_logging(prefect_logger)
        
        with KedroSession.create(project_path=project_path) as session:
            prefect_logger.info(f"Starting Kedro {self.pipeline_name} pipeline...")
            session.run(pipeline_name=self.pipeline_name)
            prefect_logger.info(f"Kedro {self.pipeline_name} pipeline completed.")
        
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

