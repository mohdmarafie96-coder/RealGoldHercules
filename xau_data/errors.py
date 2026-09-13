"""Typed exceptions for the XAUUSD data layer."""
from __future__ import annotations

__all__ = [
    "XauDataError", "NaiveDatetimeError", "ConfigError", "SchemaError",
    "DecodeError", "FetchError", "PolicyDeniedError",
]


class XauDataError(Exception):
    """Base class for every error raised by this package."""


class NaiveDatetimeError(XauDataError, ValueError):
    """A naive (timezone-less) datetime reached a boundary that requires UTC."""


class ConfigError(XauDataError, ValueError):
    """Configuration is missing, malformed, or internally inconsistent."""


class SchemaError(XauDataError, ValueError):
    """Data does not match the declared Arrow schema."""


class DecodeError(XauDataError, ValueError):
    """A source payload could not be decoded."""


class FetchError(XauDataError, RuntimeError):
    """A remote fetch failed after exhausting retries."""


class PolicyDeniedError(FetchError):
    """Egress policy refused the host. Never retried."""
