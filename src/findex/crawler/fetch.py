"""One polite HTTP GET: a hard timeout, bounded retries with backoff, Retry-After.

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        result = await fetch(client, "https://example.org/")

fetch() never raises for network or HTTP problems: it returns a FetchResult that says
what happened, because a crawler fetching thousands of pages must keep going when one
of them fails. Redirects are NOT followed by default: the target is returned in
`location`, so the crawler can check it (robots.txt, allowed domains, already seen?)
like any other link.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import httpx

from . import USER_AGENT

log = logging.getLogger("findex.crawler.fetch")

# Worth retrying: rate limiting and server-side trouble that is often temporary.
# Not retried: other 4xx (the request itself is wrong) and other statuses.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None  # None: no HTTP response at all (timeout, connection error)
    content: bytes
    content_type: str  # media type without parameters, e.g. "text/html"; "" if unknown
    location: str | None  # absolute redirect target, for 3xx responses
    elapsed: float  # seconds, including retries and the waits between them
    attempts: int
    error: str | None  # why it failed, if it did
    truncated: bool = False  # the body was longer than max_bytes and was cut
    charset: str | None = None  # from the Content-Type header, e.g. "utf-8"

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


type Sleep = Callable[[float], Awaitable[None]]


def make_client(transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """An HTTP client that identifies itself honestly on every request.

    (transport: tests pass an httpx.MockTransport -- a fake server -- here.)
    """
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        transport=transport,
        timeout=httpx.Timeout(30.0),  # a backstop; fetch() enforces its own, shorter timeout
    )


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    max_bytes: int = 5_000_000,
    follow_redirects: bool = False,
    backoff_base: float = 0.5,
    backoff_cap: float = 30.0,
    max_retry_after: float = 60.0,
    sleep: Sleep = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> FetchResult:
    """GET url. Retries timeouts, connection errors, 429 and 5xx up to max_retries times.

    The wait before retry n is the server's Retry-After if it sent one, otherwise
    exponential backoff with jitter. `sleep` and `rand` are parameters so tests can run
    without real waiting or randomness.
    """
    start = time.perf_counter()
    error = "not attempted"
    status: int | None = None
    attempt = 0
    for attempt in range(1, max_retries + 2):
        retry_after: float | None = None
        try:
            async with asyncio.timeout(timeout):  # the whole request, body included
                response, content, truncated = await _get(client, url, max_bytes, follow_redirects)
        except TimeoutError:
            status, error = None, f"timeout after {timeout:g}s"
        except (httpx.UnsupportedProtocol, httpx.InvalidURL) as exc:
            return _result(url, start, attempt, error=f"bad URL: {exc}")  # retrying can't help
        except httpx.TooManyRedirects:
            return _result(url, start, attempt, error="too many redirects")
        except httpx.TransportError as exc:  # connection refused/reset, DNS failure, ...
            status, error = None, f"{type(exc).__name__}: {exc}"
        else:
            status = response.status_code
            if status not in RETRY_STATUSES:
                if truncated:
                    log.warning("%s: body cut at %d bytes", url, max_bytes)
                return FetchResult(
                    url=url,
                    status=status,
                    content=content,
                    content_type=_media_type(response),
                    location=_location(response),
                    elapsed=time.perf_counter() - start,
                    attempts=attempt,
                    error=None if response.is_success or response.is_redirect else f"HTTP {status}",
                    truncated=truncated,
                    charset=response.charset_encoding,
                )
            error = f"HTTP {status}"
            retry_after = parse_retry_after(response.headers.get("retry-after"))

        if attempt > max_retries:
            break
        if retry_after is not None and retry_after > max_retry_after:
            log.warning(
                "%s: server asks to retry in %.0fs, more than we wait; giving up", url, retry_after
            )
            break
        delay = (
            retry_after
            if retry_after is not None
            else backoff(attempt, backoff_base, backoff_cap, rand)
        )
        log.info("%s: %s; retry %d/%d in %.2fs", url, error, attempt, max_retries, delay)
        await sleep(delay)

    log.warning("%s: failed after %d attempts: %s", url, attempt, error)
    return _result(url, start, attempt, error=error, status=status)


async def _get(
    client: httpx.AsyncClient, url: str, max_bytes: int, follow_redirects: bool
) -> tuple[httpx.Response, bytes, bool]:
    """Stream the body and stop at max_bytes: a huge file can't exhaust memory or time."""
    async with client.stream("GET", url, follow_redirects=follow_redirects) as response:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:
                return response, b"".join(chunks)[:max_bytes], True
        return response, b"".join(chunks), False


def backoff(attempt: int, base: float, cap: float, rand: Callable[[], float]) -> float:
    """Exponential backoff with jitter: base * 2**(attempt-1), capped, then randomized
    between half and all of that. The randomness spreads out retries from many clients
    (or many of our own workers) that failed at the same moment, instead of having them
    all hit the server again in lockstep."""
    ceiling = min(cap, base * 2 ** (attempt - 1))
    return ceiling / 2 + rand() * ceiling / 2


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After is either a number of seconds or an HTTP date. None if absent or invalid."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:  # HTTP dates are always GMT
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _media_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


def _location(response: httpx.Response) -> str | None:
    target = response.headers.get("location")
    return urljoin(str(response.url), target) if response.is_redirect and target else None


def _result(
    url: str, start: float, attempts: int, *, error: str, status: int | None = None
) -> FetchResult:
    return FetchResult(
        url=url,
        status=status,
        content=b"",
        content_type="",
        location=None,
        elapsed=time.perf_counter() - start,
        attempts=attempts,
        error=error,
    )
