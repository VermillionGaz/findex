"""RobotsCache, against a fake server."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable

import httpx
import pytest

from findex.crawler.fetch import make_client
from findex.crawler.robots import RobotsCache

RULES = """
User-agent: *
Disallow: /private/
Crawl-delay: 2

User-agent: findex
Disallow: /no-findex/
Crawl-delay: 5
"""


async def no_wait(seconds: float) -> None:
    pass


def check_with(handler: Callable[[httpx.Request], httpx.Response], *urls: str) -> list[bool]:
    """Ask a RobotsCache about each URL, against a fake server."""

    async def main() -> list[bool]:
        async with make_client(httpx.MockTransport(handler)) as client:
            robots = RobotsCache(client, sleep=no_wait)
            return [await robots.allowed(u) for u in urls]

    return asyncio.run(main())


def check(robots_response: httpx.Response, *urls: str) -> list[bool]:
    return check_with(lambda request: robots_response, *urls)


def test_rules_for_our_user_agent_are_obeyed() -> None:
    allowed = check(
        httpx.Response(200, text=RULES),
        "https://example.org/page",
        "https://example.org/no-findex/page",
        "https://example.org/private/page",  # only for other robots: we have our own group
    )
    assert allowed == [True, False, True]


def test_crawl_delay() -> None:
    async def main() -> float | None:
        handler = lambda request: httpx.Response(200, text=RULES)
        async with make_client(httpx.MockTransport(handler)) as client:
            return await RobotsCache(client).crawl_delay("https://example.org/")

    assert asyncio.run(main()) == 5.0


@pytest.mark.parametrize(
    ("status", "allowed"),
    [(404, True), (410, True), (401, False), (403, False), (500, False), (503, False)],
)
def test_what_a_failed_robots_fetch_means(status: int, allowed: bool) -> None:
    assert check(httpx.Response(status), "https://example.org/page") == [allowed]


def test_unreachable_site_is_off_limits() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async def main() -> bool:
        async with make_client(httpx.MockTransport(down)) as client:
            return await RobotsCache(client, sleep=no_wait).allowed("https://example.org/")

    assert asyncio.run(main()) is False


def test_robots_txt_is_fetched_once_per_site_even_when_asked_concurrently() -> None:
    fetched: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        fetched[str(request.url)] += 1
        return httpx.Response(200, text=RULES)

    async def main() -> None:
        async with make_client(httpx.MockTransport(handler)) as client:
            robots = RobotsCache(client)
            urls = [f"https://{host}/p{i}" for host in ("a.org", "b.org") for i in range(20)]
            await asyncio.gather(*(robots.allowed(u) for u in urls))  # 40 questions at once

    asyncio.run(main())
    assert fetched == {"https://a.org/robots.txt": 1, "https://b.org/robots.txt": 1}


def test_robots_txt_redirects_are_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(301, headers={"Location": "https://www.example.org/robots.txt"})
        return httpx.Response(200, text="User-agent: *\nDisallow: /\n")

    assert check_with(handler, "https://example.org/page") == [False]
