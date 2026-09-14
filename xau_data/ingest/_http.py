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
        recycle_after: int = 50,
    ) -> None:
        self._limiter = RateLimiter(requests_per_second)
        self._max_attempts = max_attempts
        self._base = backoff_base_seconds
        self._cap = backoff_max_seconds
        self._connect = connect_timeout_seconds
        self._read = read_timeout_seconds
        self._max_connections = max_connections
        self._recycle_after = max(1, recycle_after)
        self._since_recycle = 0
        self._client = self._new_client()

    def _new_client(self) -> httpx.Client:
        # Every timeout is set explicitly, POOL INCLUDED. A pool wait with no
        # deadline is how a leaked connection wedges a sequential job forever:
        # observed in the wild as 34 minutes blocked in wait_for_connection.
        return httpx.Client(
            timeout=httpx.Timeout(
                self._read, connect=self._connect, read=self._read,
                write=self._read, pool=min(30.0, self._read),
            ),
            limits=httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=self._max_connections,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )

    def _recycle(self) -> None:
        """Drop the pool and start a fresh one. Cheap insurance against leaks."""
        try:
            self._client.close()
        except Exception:
            pass
        self._client = self._new_client()
        self._since_recycle = 0

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
            if self._since_recycle >= self._recycle_after:
                self._recycle()
            self._since_recycle += 1
            r = None
            try:
                r = self._client.get(url)
                status, content = r.status_code, r.content
            except httpx.PoolTimeout as exc:
                # the pool is wedged; a fresh one is the only reliable cure
                last = f"PoolTimeout: {exc}"
                self._recycle()
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                # transport-level failures are exactly when connections leak
                self._recycle()
            else:
                if status in _DENY_STATUS:
                    raise PolicyDeniedError(
                        f"egress policy refused {url} with {status}; not retried"
                    )
                if status in (200, 404):
                    return FetchResult(status, content, attempt + 1)
                if status not in _RETRY_STATUS:
                    raise FetchError(f"{url}: unexpected status {status}")
                last = f"status {status}"
            finally:
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass
            time.sleep(self._sleep_for(attempt, None))
        raise FetchError(f"{url}: giving up after {self._max_attempts} attempts ({last})")
