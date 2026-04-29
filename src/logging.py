"""
Structured JSON logging with correlation IDs.
Each log line is a JSON object with timestamp, level, logger, message, and context.
"""
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from src.config import get_settings

# per-request correlation ID — set by middleware, included in every log line
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get current correlation ID or generate one."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    cid = cid or str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


class JSONFormatter(logging.Formatter):
    """Format every log record as a JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        # include source info for warnings and above
        if record.levelno >= logging.WARNING:
            log["file"] = record.filename
            log["line"] = record.lineno

        # include exception details if present
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        # include any extra fields passed via logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message", "module",
                "msecs", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            }:
                log[key] = value

        return json.dumps(log)


class HumanFormatter(logging.Formatter):
    """Pretty format for local development."""

    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        cid = correlation_id_var.get()
        cid_str = f"[{cid}] " if cid else ""
        timestamp = time.strftime("%H:%M:%S", time.gmtime(record.created))
        return (
            f"{color}{timestamp} {record.levelname:7}{self.RESET} "
            f"{cid_str}{record.name}: {record.getMessage()}"
        )


def setup_logging(json_format: Optional[bool] = None, level: Optional[str] = None):
    """
    Configure root logger.

    Args:
        json_format: True for JSON lines, False for human-readable.
                     Defaults to False (human) for dev, True for prod.
        level: Log level. Defaults to config.yaml setting.
    """
    settings = get_settings()
    level = level or settings.log_level
    # default: JSON in production, human in dev
    json_format = json_format if json_format is not None else False

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_format else HumanFormatter())
    root_logger.addHandler(handler)

    # quiet noisy third-party loggers
    logging.getLogger("opensearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialized — level={level}, format={'json' if json_format else 'human'}"
    )
