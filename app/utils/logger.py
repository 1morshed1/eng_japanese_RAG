# app/utils/logger.py
"""
Structured logging setup for the Healthcare RAG Assistant.

Provides JSON-formatted logging for production and human-readable
logs for development. Supports custom fields via the 'extra' parameter.

Usage:
    from app.utils.logger import setup_logging
    import logging
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(
        "Document processed",
        extra={
            "doc_id": "doc_123",
            "chunks": 15,
            "duration": 2.4
        }
    )
"""

import logging
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log records as JSON with timestamp, level, message,
    module, function, line number, and any additional fields
    passed via the 'extra' parameter in log calls.
    
    Example:
        logger.info(
            "Request completed",
            extra={
                "request_id": "abc123",
                "duration": 1.5,
                "status_code": 200
            }
        )
        
    Output:
        {
            "timestamp": "2025-11-04T12:34:56.789Z",
            "level": "INFO",
            "message": "Request completed",
            "module": "main",
            "function": "handle_request",
            "line": 45,
            "request_id": "abc123",
            "duration": 1.5,
            "status_code": 200
        }
    """
    
    # Standard logging attributes that should not be included as extra fields
    RESERVED_ATTRS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
        'pathname', 'process', 'processName', 'relativeCreated',
        'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
        'taskName'  # Added in Python 3.12
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: LogRecord instance
            
        Returns:
            JSON string representation of the log record
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add logger name if not root
        if record.name != "root":
            log_data["logger"] = record.name
        
        # Add process/thread info for debugging
        if os.getenv("LOG_PROCESS_INFO", "false").lower() == "true":
            log_data["process_id"] = record.process
            log_data["thread_id"] = record.thread
        
        # Add all custom extra fields
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    log_data[key] = str(value)
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include stack info if present
        if record.stack_info:
            log_data["stack_info"] = self.formatStack(record.stack_info)
        
        return json.dumps(log_data, default=str)


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure application logging with structured output.
    
    Sets up both console and file handlers with appropriate formatting
    based on environment (JSON for production, simple for development).
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, uses value from settings.LOG_LEVEL
    
    Returns:
        Configured root logger instance
        
    Raises:
        ValueError: If log_level is invalid
        
    Example:
        >>> from app.utils.logger import setup_logging
        >>> setup_logging("INFO")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Get logger
    logger = logging.getLogger()
    
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Get log level from settings if not provided
    if log_level is None:
        try:
            from app.config import settings
            log_level = settings.LOG_LEVEL
        except ImportError:
            log_level = "INFO"
    
    # Validate log level
    log_level = log_level.upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(
            f"Invalid log level: {log_level}. "
            "Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    
    # Set log level
    logger.setLevel(getattr(logging, log_level))
    
    # Create logs directory
    logs_dir = Path("logs")
    try:
        logs_dir.mkdir(exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create logs directory: {e}", file=sys.stderr)
    
    # ========================================
    # Console Handler
    # ========================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Determine environment
    try:
        from app.config import settings
        env = settings.ENV
    except (ImportError, AttributeError):
        env = os.getenv("ENV", "development")
    
    # Use JSON format in production, simple format in development
    if env == "production":
        console_handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        console_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
    
    logger.addHandler(console_handler)
    
    # ========================================
    # File Handler (with rotation)
    # ========================================
    try:
        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Always use JSON format for file logs (easy to parse)
        file_handler.setFormatter(JSONFormatter())
        
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"Could not set up file logging: {e}")
    
    # ========================================
    # Configure third-party loggers
    # ========================================
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Log startup message
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "environment": env,
            "console_format": "json" if env == "production" else "simple",
            "file_logging": True
        }
    )
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Convenience function for getting module-specific loggers.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from app.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


# ========================================
# Context Managers for Request Logging
# ========================================

from contextlib import contextmanager
import time

@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, **extra_fields):
    """
    Context manager to log execution time of an operation.
    
    Args:
        logger: Logger instance to use
        operation: Description of the operation
        **extra_fields: Additional fields to include in log
        
    Example:
        >>> logger = get_logger(__name__)
        >>> with log_execution_time(logger, "document_processing", doc_id="doc_123"):
        >>>     process_document()
    """
    start_time = time.time()
    
    logger.info(
        f"{operation} started",
        extra=extra_fields
    )
    
    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"{operation} failed",
            extra={
                **extra_fields,
                "duration_seconds": round(duration, 3),
                "error": str(e)
            },
            exc_info=True
        )
        raise
    else:
        duration = time.time() - start_time
        logger.info(
            f"{operation} completed",
            extra={
                **extra_fields,
                "duration_seconds": round(duration, 3)
            }
        )