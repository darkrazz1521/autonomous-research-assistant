"""Retry helpers for sync and async I/O tasks."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from autonomous_research_assistant_data.config import RetryConfig

T = TypeVar("T")


def _sleep_seconds(attempt: int, config: RetryConfig) -> float:
    delay = min(
        config.base_delay_seconds * (config.backoff_multiplier ** max(attempt - 1, 0)),
        config.max_delay_seconds,
    )
    if config.jitter_seconds > 0:
        delay += random.uniform(0, config.jitter_seconds)
    return delay


async def retry_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Run an async function with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func()
        except retryable_exceptions as exc:
            last_error = exc
            if attempt >= config.max_attempts:
                break
            await asyncio.sleep(_sleep_seconds(attempt, config))
    assert last_error is not None
    raise last_error


def retry_sync(
    func: Callable[[], T],
    config: RetryConfig,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Run a sync function with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            last_error = exc
            if attempt >= config.max_attempts:
                break
            time.sleep(_sleep_seconds(attempt, config))
    assert last_error is not None
    raise last_error

