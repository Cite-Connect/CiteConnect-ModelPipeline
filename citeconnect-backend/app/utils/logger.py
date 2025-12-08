"""
Structured logging configuration for CiteConnect.
Provides detailed logging with context for debugging.
Outputs to both Console (Pretty) and File (JSON Lines).
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Any
import structlog
from app.config import settings

def setup_logging() -> None:
    """
    Configure structured logging with context processors.
    Logs include: timestamp, level, logger name, function, line number, and message.
    """
    
    # 1. Define processors used by BOTH console and file
    shared_processors = [
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
    ]

    # 2. Configure structlog to wrap data for the standard library
    structlog.configure(
        processors=shared_processors + [
            # This prepares the log entry for standard logging handlers
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 3. Create the 'logs' directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 4. Define specific formatters for Console vs File
    
    # Console: Use readable colors if DEBUG is True, otherwise use JSON
    # This is useful if you view production logs via Docker logs/stdout
    console_renderer = (
        structlog.dev.ConsoleRenderer() 
        if settings.DEBUG 
        else structlog.processors.JSONRenderer()
    )
    
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_renderer,
        foreign_pre_chain=shared_processors,
    )

    # File: ALWAYS use JSON (clean, parsable, no colors)
    # sort_keys=True ensures consistent field order for easier reading
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(sort_keys=False),
        foreign_pre_chain=shared_processors,
    )

    # 5. Configure Standard Library Handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Clear existing handlers to prevent duplicates during reloads
    root_logger.handlers = []

    # -- Handler A: Console (Stdout) --
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # -- Handler B: File (Rotating) --
    # Writes to logs/citeconnect.log
    # maxBytes=10MB, backupCount=5 (keeps last 5 files)
    file_handler = RotatingFileHandler(
        "logs/citeconnect.log", 
        maxBytes=10 * 1024 * 1024, 
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


def log_function_entry(logger: structlog.stdlib.BoundLogger, **kwargs: Any) -> None:
    """
    Log function entry with parameters.
    """
    logger.debug("Function entry", **kwargs)


def log_function_exit(
    logger: structlog.stdlib.BoundLogger, 
    result: Any = None,
    **kwargs: Any
) -> None:
    """
    Log function exit with result.
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
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {})
    }
    logger.error("Error occurred", **error_data, exc_info=True)