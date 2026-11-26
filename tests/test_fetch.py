"""fetch(), against a fake server (httpx.MockTransport): no network, no real waiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx
import pytest

from findex.crawler import USER_AGENT
from findex.crawler.fetch import FetchResult, backoff, fetch, make_client, parse_retry_after

# A fake server: request in, response out (async handlers are allowed too).
type Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


def run_fetch(
    handler: Handler, url: str = "https://example.org/page", **kwargs: float
) -> tuple[FetchResult, list[float]]:
    """fetch() once against `handler`; returns the result and every wait it asked for."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    async def main() -> FetchResult:
        async with make_client(httpx.MockTransport(handler)) as client:
            return await fetch(client, url, sleep=fake_sleep, rand=lambda: 1.0, **kwargs)

    return asyncio.run(main()), waits


def responses(*items: httpx.Response | Exception) -> Handler:
    """A fake server that answers with these, in order (an exception = network failure)."""
    queue = list(items)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def test_success() -> None:
    page = httpx.Response(200, html="<p>hello</p>")
    result, waits = run_fetch(responses(page))
    assert result.ok and result.status == 200 and result.error is None
    assert result.content == b"<p>hello</p>" and result.content_type == "text/html"
    assert result.attempts == 1 and waits == []


def test_sends_an_honest_user_agent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200)

    run_fetch(handler)
    assert seen == [USER_AGENT] and "+https://" in USER_AGENT  # with a contact URL


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_client_errors_are_not_retried(status: int) -> None:
    result, waits = run_fetch(responses(httpx.Response(status)))
    assert result.status == status and result.error == f"HTTP {status}"
    assert result.attempts == 1 and waits == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_rate_limits_and_server_errors_are_retried(status: int) -> None:
    result, waits = run_fetch(responses(httpx.Response(status), httpx.Response(200)))
    assert result.ok and result.attempts == 2 and len(waits) == 1


def test_gives_up_after_max_retries_with_growing_waits() -> None:
    result, waits = run_fetch(responses(*[httpx.Response(503)] * 4), max_retries=3)
    assert not result.ok and result.status == 503 and result.attempts == 4
    assert waits == [0.5, 1.0, 2.0]  # base 0.5 doubling, rand() = 1.0: the top of each range


def test_retry_after_seconds_is_honoured() -> None:
    limited = httpx.Response(429, headers={"Retry-After": "7"})
    result, waits = run_fetch(responses(limited, httpx.Response(200)))
    assert result.ok and waits == [7.0]


def test_retry_after_too_long_means_give_up() -> None:
    limited = httpx.Response(429, headers={"Retry-After": "3600"})
    result, waits = run_fetch(responses(limited), max_retry_after=60)
    assert result.status == 429 and result.attempts == 1 and waits == []


def test_timeouts_are_retried() -> None:
    calls = 0

    async def slow_then_fast(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)  # longer than the timeout below
        return httpx.Response(200)

    result, waits = run_fetch(slow_then_fast, timeout=0.05)
    assert result.ok and result.attempts == 2 and len(waits) == 1


def test_connection_errors_are_retried_then_reported() -> None:
    down = httpx.ConnectError("connection refused")
    result, waits = run_fetch(responses(down, down), max_retries=1)
    assert result.status is None and result.attempts == 2 and len(waits) == 1
    assert result.error is not None and "ConnectError" in result.error


def test_unsupported_url_is_not_retried() -> None:
    # The real transport refuses a mailto: URL; the fake server raises what it would.
    refused = httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'mailto:'")
    result, waits = run_fetch(responses(refused), url="mailto:someone@example.org")
    assert result.error is not None and result.error.startswith("bad URL") and waits == []


def test_redirects_are_reported_not_followed() -> None:
    moved = httpx.Response(301, headers={"Location": "/new-home"})
    result, _ = run_fetch(responses(moved), url="https://example.org/old")
    assert result.status == 301 and result.error is None and not result.ok
    assert result.location == "https://example.org/new-home"  # made absolute


def test_huge_bodies_are_cut() -> None:
    result, _ = run_fetch(responses(httpx.Response(200, content=b"x" * 10_000)), max_bytes=1000)
    assert result.truncated and len(result.content) == 1000


@pytest.mark.parametrize("attempt", [1, 2, 3, 10])
def test_backoff_stays_in_its_range(attempt: int) -> None:
    ceiling = min(30.0, 0.5 * 2 ** (attempt - 1))
    low = backoff(attempt, 0.5, 30.0, rand=lambda: 0.0)
    high = backoff(attempt, 0.5, 30.0, rand=lambda: 1.0)
    assert (low, high) == (ceiling / 2, ceiling)


def test_parse_retry_after() -> None:
    now = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after(" 5 ") == 5.0
    assert parse_retry_after("Thu, 26 Nov 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Thu, 26 Nov 2026 11:00:00 GMT", now=now) == 0.0  # in the past
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
