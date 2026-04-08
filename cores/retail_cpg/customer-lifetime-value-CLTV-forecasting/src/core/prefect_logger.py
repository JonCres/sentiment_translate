import logging


class PrefectLogHandler(logging.Handler):
    """Custom logging handler that forwards logs to Prefect's logger"""
    
    def __init__(self, prefect_logger):
        super().__init__()
        self.prefect_logger = prefect_logger
    
    def emit(self, record):
        """Emit a log record to Prefect's logger"""
        try:
            msg = self.format(record)
            # Map logging levels to Prefect logger methods
            if record.levelno >= logging.ERROR:
                self.prefect_logger.error(msg)
            elif record.levelno >= logging.WARNING:
                self.prefect_logger.warning(msg)
            elif record.levelno >= logging.INFO:
                self.prefect_logger.info(msg)
            else:
                self.prefect_logger.debug(msg)
        except Exception:
            self.handleError(record)