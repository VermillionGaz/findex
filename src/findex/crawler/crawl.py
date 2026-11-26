"""The crawler: many polite fetches at once, in one thread, with asyncio.

    async with aclosing(crawl(["https://docs.example.org/"], max_pages=200)) as pages:
        async for page in pages:
            ...

How it fits together:
  * frontier: an asyncio.Queue of URLs still to visit. A URL is added to `seen` when it
    is queued, so nothing is queued -- or fetched -- twice.
  * workers: `concurrency` tasks in a TaskGroup, each looping: take a URL, check it
    against robots.txt, wait for its host's turn, fetch, parse, queue the new links.
  * limits: a global semaphore (at most `concurrency` requests in flight) and, per host,
    a semaphore (`per_host` at once) plus a minimum delay between request starts --
    `min_delay`, or the site's Crawl-delay if that is longer.
  * output: pages go through a second queue to the async generator, which yields them.
    The TaskGroup lives in one "runner" task owned by the generator: yielding from
    inside a TaskGroup would break cancellation when the caller stops iterating early.
    Whatever happens -- max_pages reached, an error, the caller breaking out of its
    loop -- the generator's `finally` cancels the runner and waits for it, so no task
    outlives the crawl and the HTTP client is always closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import os
import time
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx

from .fetch import FetchResult, fetch, make_client
from .html import ParsedPage, parse_html
from .robots import RobotsCache
from .urls import canonicalize, host_of, in_domains

log = logging.getLogger("findex.crawler.crawl")

HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})

# Parsing processes: page parsing takes milliseconds while fetching takes hundreds of
# them, so a few processes keep up with many concurrent fetches.
PARSE_WORKERS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: datetime


@dataclass(slots=True)
class CrawlStats:
    """Live counters, for a progress display. Updated by the crawler as it goes."""

    pages: int = 0  # pages yielded
    fetched: int = 0  # HTTP responses received (any status)
    errors: int = 0  # failed fetches (no response, 4xx, 5xx after retries)
    skipped: int = 0  # blocked by robots.txt, not HTML, noindex
    in_flight: int = 0  # requests currently waiting for a response
    queued: int = 0  # URLs in the frontier
    last_url: str = ""  # the most recent page yielded
    started: float = field(default_factory=time.monotonic)

    @property
    def pages_per_second(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.pages / elapsed if elapsed > 0 else 0.0


class _Host:
    """Politeness for one host: at most `per_host` requests at once, and request
    starts at least `delay` seconds apart."""

    def __init__(self, per_host: int) -> None:
        self.slots = asyncio.Semaphore(per_host)
        self.turn = asyncio.Lock()
        self.next_start = 0.0

    async def wait_turn(self, delay: float) -> None:
        async with self.turn:  # one request at a time takes its place in the schedule
            loop = asyncio.get_running_loop()
            wait = self.next_start - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self.next_start = loop.time() + delay


_DONE = object()  # end-of-crawl marker on the output queue


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Iterable[str] | None = None,
    min_delay: float = 1.0,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
    executor: Executor | None = None,
    stats: CrawlStats | None = None,
) -> AsyncGenerator[Page]:
    """Crawl from the seeds and yield up to max_pages pages, as they are fetched.

    allowed_domains defaults to the seeds' hosts (subdomains included). Pass a client
    to use your own (tests pass one with a fake server); otherwise one is created and
    closed here. Likewise `executor`, which parses pages: by default a small process
    pool is started for this crawl and shut down at its end.
    """
    seed_urls = [u for u in (canonicalize(s) for s in seeds) if u]
    if not seed_urls:
        raise ValueError("no valid http(s) seed URLs")
    domains = frozenset(d.lower() for d in allowed_domains or (host_of(u) for u in seed_urls))
    own_pool = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(PARSE_WORKERS, mp_context=mp.get_context("spawn"))
    crawler = _Crawler(
        max_pages,
        concurrency,
        per_host,
        domains,
        min_delay,
        timeout,
        executor,
        stats or CrawlStats(),
    )

    runner = asyncio.create_task(crawler.run(seed_urls, client))
    try:
        while (item := await crawler.out.get()) is not _DONE:
            assert isinstance(item, Page)
            yield item
    except BaseException:  # the caller stopped iterating early (GeneratorExit), or was cancelled
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner  # wait until every worker has stopped and the client is closed
        raise
    finally:
        if own_pool:  # shutting down waits for the processes to exit: in a thread, not here
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
    await runner  # finished normally: re-raises anything that went wrong inside the crawl


class _Crawler:
    def __init__(
        self,
        max_pages: int,
        concurrency: int,
        per_host: int,
        domains: frozenset[str],
        min_delay: float,
        timeout: float,
        executor: Executor,
        stats: CrawlStats,
    ) -> None:
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.per_host = per_host
        self.domains = domains
        self.min_delay = min_delay
        self.timeout = timeout
        self.executor = executor
        self.stats = stats
        self.frontier: asyncio.Queue[str] = asyncio.Queue()
        self.out: asyncio.Queue[Page | object] = asyncio.Queue()
        self.seen: set[str] = set()
        self.hosts: dict[str, _Host] = {}
        self.requests = asyncio.Semaphore(concurrency)  # global limit on requests in flight
        self.finished = asyncio.Event()  # set when max_pages pages have been produced

    async def run(self, seeds: list[str], client: httpx.AsyncClient | None) -> None:
        try:
            async with (
                make_client() if client is None else _borrowed(client) as http,
                asyncio.TaskGroup() as tg,
            ):
                self.robots = RobotsCache(http)
                # Start the parsing processes now, from a thread: starting a process blocks,
                # and if the first parse started them, the event loop would stall meanwhile.
                await asyncio.to_thread(_warm_up, self.executor)
                for url in seeds:
                    self._enqueue(url)
                workers = [tg.create_task(self._worker(http)) for _ in range(self.concurrency)]
                # Stop when the frontier is exhausted or the page budget is reached,
                # whichever comes first; then cancel the workers (they loop forever).
                frontier_done = tg.create_task(self.frontier.join())
                budget_done = tg.create_task(self.finished.wait())
                await asyncio.wait(
                    {frontier_done, budget_done}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in (*workers, frontier_done, budget_done):
                    task.cancel()
        finally:
            self.out.put_nowait(_DONE)

    def _enqueue(self, url: str) -> None:
        if url in self.seen or not in_domains(url, self.domains):
            return
        self.seen.add(url)
        self.frontier.put_nowait(url)
        self.stats.queued = self.frontier.qsize()

    async def _worker(self, http: httpx.AsyncClient) -> None:
        while True:
            url = await self.frontier.get()
            self.stats.queued = self.frontier.qsize()
            try:
                await self._visit(http, url)
            finally:
                self.frontier.task_done()

    async def _visit(self, http: httpx.AsyncClient, url: str) -> None:
        if self.finished.is_set():
            return
        if not await self.robots.allowed(url):
            self.stats.skipped += 1
            log.info("robots.txt disallows %s", url)
            return
        result = await self._polite_fetch(http, url)

        # /docs -> /docs/ canonicalizes back to the URL we just fetched, so the frontier
        # would skip it as already seen: follow that one redirect here instead.
        if result.location and canonicalize(result.location) == url:
            result = await self._polite_fetch(http, result.location)
        if result.location:  # any other redirect: the target is just another link
            target = canonicalize(result.location)
            if target:
                self._enqueue(target)
            log.info("%s %s -> %s", result.status, url, result.location)
            return
        if not result.ok:
            self.stats.errors += 1
            # A 404 is an everyday event in crawling: log file only. Failures that may mean
            # trouble -- no response at all, or 5xx after every retry -- also show on screen.
            serious = result.status is None or result.status >= 500
            log.log(
                logging.WARNING if serious else logging.INFO,
                "%s %s (%s, %d attempts)",
                result.status or "---",
                url,
                result.error,
                result.attempts,
            )
            return
        if result.content_type not in HTML_TYPES:
            self.stats.skipped += 1
            log.info(
                "%s %s skipped: %s", result.status, url, result.content_type or "no content type"
            )
            return

        # Decoding, parsing and canonicalizing links are CPU work: done in another process,
        # which has its own GIL. (A thread would not do: it shares this thread's GIL, so the
        # event loop would stall whenever the parser holds it -- Lab 5's lesson again.)
        # The loop only does what's cheap: set lookups and queue insertions.
        page, links = await asyncio.get_running_loop().run_in_executor(
            self.executor, _digest, result.content, result.charset, url
        )
        for target in links:
            self._enqueue(target)
        if page.noindex:
            self.stats.skipped += 1
            log.info("%s %s skipped: noindex", result.status, url)
            return
        self._emit(Page(url, page.title, page.text, datetime.now(UTC)), result)

    async def _polite_fetch(self, http: httpx.AsyncClient, url: str) -> FetchResult:
        host = self.hosts.setdefault(host_of(url), _Host(self.per_host))
        delay = max(self.min_delay, await self.robots.crawl_delay(url) or 0.0)
        async with host.slots:
            await host.wait_turn(delay)  # may sleep: not holding a global slot meanwhile
            async with self.requests:
                self.stats.in_flight += 1
                try:
                    result = await fetch(http, url, timeout=self.timeout)
                finally:
                    self.stats.in_flight -= 1
        self.stats.fetched += 1
        return result

    def _emit(self, page: Page, result: FetchResult) -> None:
        # No await between the check and the increment: in asyncio nothing else can run
        # in between, so two workers can't both take the last page slot.
        if self.stats.pages >= self.max_pages:
            return
        self.stats.pages += 1
        self.stats.last_url = page.url
        log.info(
            "%s %s (%d bytes, %.2fs)", result.status, page.url, len(result.content), result.elapsed
        )
        self.out.put_nowait(page)
        if self.stats.pages >= self.max_pages:
            self.finished.set()


def _warm_up(executor: Executor) -> None:
    """Make a process pool start its workers (one trivial task each), and wait until it has."""
    for future in [executor.submit(int) for _ in range(PARSE_WORKERS)]:
        future.result()


def _digest(content: bytes, charset: str | None, url: str) -> tuple[ParsedPage, list[str]]:
    """The CPU-heavy part of handling a page, run in a worker process: the parsed page, and
    its links in canonical form, duplicates removed, in page order. Module-level, with
    small picklable arguments, so a process pool can run it."""
    html = content.decode(charset or "utf-8", errors="replace")
    page = parse_html(html, url)
    links = dict.fromkeys(t for t in (canonicalize(link) for link in page.links) if t)
    return page, list(links)


class _borrowed:
    """`async with` for a client someone else owns: use it, but don't close it."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, *exc: object) -> None:
        pass


async def crawl_to_jsonl(path: Path, seeds: Iterable[str], **options: Any) -> int:
    """Crawl, writing each page to `path` as one JSON line the moment it arrives: the
    format `findex index` reads. Returns the number of pages written.

    Pages are never buffered, so an interrupted crawl keeps everything fetched so far.
    The writes go through a thread: even a slow disk can't stall the event loop.
    """
    written = 0
    with path.open("w", encoding="utf-8") as f:
        async with contextlib.aclosing(crawl(seeds, **options)) as pages:
            async for page in pages:
                line = json.dumps(
                    {
                        "id": page.url,
                        "url": page.url,
                        "title": page.title or page.url,
                        "text": page.text,
                        "fetched_at": page.fetched_at.isoformat(),
                    },
                    ensure_ascii=False,
                )
                await asyncio.to_thread(_write_line, f, line)
                written += 1
    return written


def _write_line(f: TextIO, line: str) -> None:
    f.write(line + "\n")
    f.flush()  # each page is on disk as soon as it's written, not when the crawl ends
