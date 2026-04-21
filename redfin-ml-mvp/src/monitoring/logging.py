"""Structured JSON logging — one log line per request, one per prediction.

Shape matters: every log line is JSON with a stable schema so downstream
(Datadog/Splunk) can build dashboards and SLOs on top. Request-ID correlation
ties API logs to downstream feature-store logs.
"""
from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "redfin-ml") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
