"""Centralized structured (JSON) logging for MACLLS.

Emits **one JSON object per line to stdout** so Docker / OCI / any log shipper can
capture the multi-agent pipeline's activity without extra configuration. Uses only
the standard library (no `python-json-logger` dependency).

Usage — call once at each entry point (Streamlit `app.py`, `cli.py`):

    from observability.logger_setup import configure_logging
    configure_logging()               # level from LOG_LEVEL env, default INFO

    import logging
    logging.getLogger(__name__).info("agent.done", extra={"agent": "L1", "duration_ms": 812})

`configure_logging()` is idempotent — safe to call on every Streamlit rerun.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys

from observability.correlation import CorrelationIdFilter, get_correlation_id

# Standard `LogRecord` attributes — anything NOT in here (and not private) is treated
# as a caller-supplied `extra={...}` field and included in the JSON payload.
_RESERVED_ATTRS = frozenset(
    {
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg",
        "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName", "taskName", "correlation_id",
    }
)

# Third-party loggers that are chatty at INFO (HTTP request lines, the google-genai
# "AFC is enabled with max remote calls" notice, etc.). Kept at WARNING so the JSON
# stream stays about *our* pipeline.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "google", "google_genai", "spacy")

# Process-level guard: keeps configure_logging() idempotent across module re-imports
# (Streamlit re-executes the script — and thus re-imports — on every interaction).
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single-line JSON object.

    Always emits: ``timestamp`` (UTC, ISO-8601, ms), ``level``, ``logger``,
    ``message``, ``correlation_id``. Merges any structured ``extra={...}`` fields
    (e.g. ``agent``, ``duration_ms``, ``token_count``) and attaches a formatted
    ``error`` string when the record carries exception info.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # Filter sets this; fall back to a live read so a stray handler still works.
            "correlation_id": getattr(record, "correlation_id", get_correlation_id()),
        }

        # Merge caller-supplied context passed via logger.<level>(msg, extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        # default=str keeps non-JSON-native values (enums, datetimes, objects) safe.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON stdout handler on the root logger. Idempotent.

    Log level resolves in order: the ``level`` argument → the ``LOG_LEVEL``
    environment variable → ``INFO``. Re-calling only re-syncs the level (never
    stacks duplicate handlers).
    """
    global _CONFIGURED

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if _CONFIGURED:
        return  # already installed this process; just keep the level in sync above

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(CorrelationIdFilter())

    # Replace any pre-existing handlers (e.g. Streamlit's default) so the stream
    # stays pure line-delimited JSON.
    root.handlers.clear()
    root.addHandler(handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def reset_logging_for_tests() -> None:
    """Test-only helper: clear the idempotency guard and root handlers so a test can
    reconfigure logging from a clean slate."""
    global _CONFIGURED
    _CONFIGURED = False
    logging.getLogger().handlers.clear()
