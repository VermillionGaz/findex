"""Diagnostic: which event-loop steps are slow during a crawl of one big page?

    uv run python benchmarks/loop_probe.py

Crawls a fake site (no network) with asyncio's debug mode reporting every step that
holds the event loop for more than 5 ms, and prints them with their duration and the
line the task was at. Used to track down a slow step seen only on Windows.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import sys
import time
from contextlib import aclosing

import httpx

from findex.crawler.crawl import crawl
from findex.crawler.fetch import make_client
from findex.crawler.html import parse_html

SITE = "https://site.org/"
LINKS = "".join(f'<a href="/p{i}">link</a>' for i in range(300))
BIG = f"<html><head><title>Big</title></head><body>{'<p>word </p>' * 40_000}{LINKS}</body></html>"
BODY = BIG.encode()


class Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def handler(request: httpx.Request) -> httpx.Response:
    if str(request.url) == SITE:
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=BODY)
    return httpx.Response(404)


async def main() -> None:
    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = 0.005
    print(f"event loop: {type(loop).__name__}")
    async with (
        make_client(httpx.MockTransport(handler)) as client,
        aclosing(crawl([SITE], client=client, min_delay=0, max_pages=1)) as pages,
    ):
        async for _ in pages:
            pass


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    t0 = time.perf_counter()
    parse_html(BIG, SITE)
    print(f"parse_html on the big page: {time.perf_counter() - t0:.3f}s")
    collect = Collect()
    logging.getLogger("asyncio").addHandler(collect)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    t0 = time.perf_counter()
    asyncio.run(main(), debug=True)
    print(f"whole crawl: {time.perf_counter() - t0:.3f}s")
    print("event-loop steps over 5 ms:")
    for message in collect.messages:
        took = re.search(r"took ([0-9.]+) seconds", message)
        where = re.findall(r"running at (\S+?:\d+)", message)
        what = re.search(r"Executing <(\w+)", message)
        if took:
            print(
                f"  {float(took.group(1)) * 1000:7.1f} ms  {what.group(1) if what else '?'}  at {where}"
            )
            print(f"            {message[:300]}")
