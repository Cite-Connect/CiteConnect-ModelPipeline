# app/core/logging.py

"""
Logging Configuration Module

This module sets up structured logging for the CiteConnect application.
It provides JSON-formatted logs for production and human-readable logs
for development.

Features:
- JSON formatting for production (machine-readable)
- Colored console output for development (human-readable)
- Request ID tracking across log messages
- Log level configuration via environment variables
- Automatic log rotation for file handlers

Usage:
    from app.core.logging import setup_logging, get_logger
    
    # Setup logging once at application startup
    setup_logging()
    
    # Get logger for your module
    logger = get_logger(__name__)
    logger.info("Application started")
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log records as JSON objects for easy parsing by log aggregation
    systems like ELK Stack, Splunk, or CloudWatch.
    
    Output format:
        {
            "timestamp": "2025-11-10T10:30:45.123456",
            "level": "INFO",
            "logger": "app.services.search_service",
            "message": "Search completed successfully",
            "module": "search_service",
            "function": "search",
            "line": 145,
            "request_id": "abc-123-def",
            "user_id": 12345,
            "duration_ms": 234
        }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.
        
        Args:
            record: LogRecord object to format
        
        Returns:
            JSON-formatted string representation of log record
        """
        # Base log data
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        # These are added via logger.info("msg", extra={...})
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'status_code'):
            log_data['status_code'] = record.status_code
        
        if hasattr(record, 'method'):
            log_data['method'] = record.method
        
        if hasattr(record, 'path'):
            log_data['path'] = record.path
        
        if hasattr(record, 'client_ip'):
            log_data['client_ip'] = record.client_ip
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.exc_info:
            log_data['stack_trace'] = ''.join(
                traceback.format_exception(*record.exc_info)
            )
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    
    Adds ANSI color codes to log output for better readability in terminal.
    Different log levels get different colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red + Bold
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: LogRecord object to format
        
        Returns:
            Colored string representation of log record
        """
        # Get color for this log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build colored log message
        log_message = (
            f"{color}[{timestamp}] "
            f"{record.levelname:8s}{reset} "
            f"{record.name}:{record.lineno} - "
            f"{record.getMessage()}"
        )
        
        # Add extra fields if present
        extras = []
        if hasattr(record, 'request_id'):
            extras.append(f"request_id={record.request_id}")
        if hasattr(record, 'user_id'):
            extras.append(f"user_id={record.user_id}")
        if hasattr(record, 'duration_ms'):
            extras.append(f"duration={record.duration_ms}ms")
        
        if extras:
            log_message += f" [{', '.join(extras)}]"
        
        # Add exception info if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Sets up appropriate formatters and handlers based on environment.
    - Development: Colored console output
    - Production: JSON-formatted logs
    
    Args:
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: Environment name ("development" or "production")
        log_file: Optional path to log file. If provided, logs are written to file.
    
    Example:
        >>> setup_logging(log_level="DEBUG", environment="development")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Choose formatter based on environment
    if environment.lower() == "production":
        # JSON formatter for production (machine-readable)
        console_formatter = JSONFormatter()
    else:
        # Colored formatter for development (human-readable)
        console_formatter = ColoredFormatter()
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler always uses JSON format for easy parsing
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Set log levels for external libraries to reduce noise
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('weaviate').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Log that logging is configured
    root_logger.info(
        f"Logging configured successfully",
        extra={
            "log_level": log_level,
            "environment": environment,
            "log_file": log_file
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This is a convenience function that wraps logging.getLogger()
    and ensures consistent logger naming across the application.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
    
    Returns:
        Logger instance configured with application settings
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting search operation", extra={"query": "machine learning"})
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to all log messages.
    
    Useful for adding request-specific context (request_id, user_id)
    to all logs within a request handler.
    
    Example:
        >>> base_logger = get_logger(__name__)
        >>> logger = LoggerAdapter(base_logger, {"request_id": "abc-123", "user_id": 456})
        >>> logger.info("Processing request")
        # Output will include request_id and user_id automatically
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message and add contextual information.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments for logging
        
        Returns:
            Tuple of (message, kwargs) with added context
        """
        # Add extra fields from adapter
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


def log_function_call(func):
    """
    Decorator to automatically log function entry and exit.
    
    Logs function name, arguments, and execution time.
    Useful for debugging and performance monitoring.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with logging
    
    Example:
        >>> @log_function_call
        >>> def search_papers(query: str, limit: int = 10):
        >>>     # Function implementation
        >>>     pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(
            f"Entering function: {func.__name__}",
            extra={
                "function": func.__name__,
                "args": str(args)[:100],  # Truncate long arguments
                "kwargs": str(kwargs)[:100]
            }
        )
        
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful completion
            logger.debug(
                f"Exiting function: {func.__name__}",
                extra={
                    "function": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                    "success": False
                },
                exc_info=True
            )
            
            # Re-raise exception
            raise
    
    return wrapper


# Initialize module logger
logger = get_logger(__name__)
logger.info("Logging module loaded successfully")