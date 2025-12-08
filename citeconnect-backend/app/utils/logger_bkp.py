"""
Structured logging configuration for CiteConnect.
Provides detailed logging with context for debugging.
"""
import logging
import sys
from typing import Any
import structlog
from app.config import settings


def setup_logging() -> None:
    """
    Configure structured logging with context processors.
    Logs include: timestamp, level, logger name, function, line number, and message.
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.dev.ConsoleRenderer() if settings.DEBUG 
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        BoundLogger: Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Starting process", user_id=123, action="recommendation")
    """
    return structlog.get_logger(name)


def log_function_entry(logger: structlog.stdlib.BoundLogger, **kwargs: Any) -> None:
    """
    Log function entry with parameters.
    
    Args:
        logger: Logger instance
        **kwargs: Function parameters to log
    """
    logger.debug("Function entry", **kwargs)


def log_function_exit(
    logger: structlog.stdlib.BoundLogger, 
    result: Any = None,
    **kwargs: Any
) -> None:
    """
    Log function exit with result.
    
    Args:
        logger: Logger instance
        result: Function return value
        **kwargs: Additional context
    """
    log_data = {"result_type": type(result).__name__, **kwargs}
    logger.debug("Function exit", **log_data)


def log_error(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    context: dict[str, Any] | None = None
) -> None:
    """
    Log error with full context and stack trace.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about the error
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {})
    }
    logger.error("Error occurred", **error_data, exc_info=True)