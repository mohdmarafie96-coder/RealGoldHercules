"""Shared rate-limited, retrying HTTP client."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Final

import httpx

from ..errors import FetchError, PolicyDeniedError

__all__ = ["RateLimiter", "HttpFetcher", "FetchResult"]

_RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})
_DENY_STATUS: Final[frozenset[int]] = frozenset({403, 407})


class RateLimiter:
    """Token bucket. Thread-safe, shared across workers."""

    __slots__ = ("_rate", "_capacity", "_tokens", "_last", "_lock")

    def __init__(self, per_second: float, capacity: float | None = None) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self._rate = per_second
        self._capacity = capacity if capacity is not None else max(1.0, per_second)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


@dataclass(frozen=True, slots=True)
class FetchResult:
    status: int
    content: bytes
    attempts: int


class HttpFetcher:
    """GET with a token bucket, exponential backoff and full jitter.

    404 is returned to the caller rather than raised: for Dukascopy an absent
    hourly file is information, not an error. 403/407 are egress policy denials
    and are never retried.
    """

    def __init__(
        self,
        *,
        requests_per_second: float,
        max_attempts: int,
        backoff_base_seconds: float,
        backoff_max_seconds: float,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_connections: int = 2,
    ) -> None:
        self._limiter = RateLimiter(requests_per_second)
        self._max_attempts = max_attempts
        self._base = backoff_base_seconds
        self._cap = backoff_max_seconds
        self._client = httpx.Client(
            timeout=httpx.Timeout(read_timeout_seconds, connect=connect_timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=90.0,
            ),
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _sleep_for(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self._cap, float(retry_after))
            except ValueError:
                pass
        return min(self._cap, self._base * (2 ** attempt)) * (0.5 + random.random())

    def get(self, url: str) -> FetchResult:
        last: str = "no attempt made"
        for attempt in range(self._max_attempts):
            self._limiter.acquire()
            try:
                r = self._client.get(url)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if r.status_code in _DENY_STATUS:
                    raise PolicyDeniedError(
                        f"egress policy refused {url} with {r.status_code}; not retried"
                    )
                if r.status_code in (200, 404):
                    return FetchResult(r.status_code, r.content, attempt + 1)
                if r.status_code not in _RETRY_STATUS:
                    raise FetchError(f"{url}: unexpected status {r.status_code}")
                last = f"status {r.status_code}"
            time.sleep(self._sleep_for(attempt, None))
        raise FetchError(f"{url}: giving up after {self._max_attempts} attempts ({last})")
