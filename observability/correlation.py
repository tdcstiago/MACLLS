"""Correlation-ID propagation for tracing a single request across the whole
multi-agent pipeline (orchestrator → spaCy MCP tools → LLM calls → cache).

The ID lives in a ``ContextVar`` so it flows *implicitly* through the call stack
without threading an ``id`` parameter through domain function signatures — which
keeps the layer boundaries clean. For worker threads (the L1/L2 specialists run
in a ``ThreadPoolExecutor``), the caller must copy the context with
``contextvars.copy_context()`` because ContextVars do NOT auto-propagate into
threads it did not itself spawn.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import uuid

# Value reported when no request scope is active (module import, idle worker, …).
NO_CORRELATION_ID = "-"

# The single source of truth for "which request am I currently serving?".
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=NO_CORRELATION_ID
)


def new_correlation_id() -> str:
    """Return a fresh, unique correlation id (uuid4 hex — 32 chars, no dashes)."""
    return uuid.uuid4().hex


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation id to the current context (generating one if omitted).

    Call this once per request at the entry point (Streamlit submit / CLI ``main``).
    Returns the id that was set so the caller can log or echo it.
    """
    cid = correlation_id or new_correlation_id()
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context, or ``"-"``."""
    return _correlation_id.get()


@contextlib.contextmanager
def correlation_scope(correlation_id: str | None = None):
    """Bind a correlation id for the duration of a ``with`` block, then restore the
    previous value. Handy for tests and for discrete background tasks.
    """
    token = _correlation_id.set(correlation_id or new_correlation_id())
    try:
        yield _correlation_id.get()
    finally:
        _correlation_id.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Stamps the current correlation id onto every ``LogRecord`` as
    ``record.correlation_id`` so handlers/formatters can emit it. Attached to the
    stdout handler in ``logger_setup.configure_logging``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True
