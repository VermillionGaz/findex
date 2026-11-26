"""The crawler, against fake websites served by httpx.MockTransport: no network."""

from __future__ import annotations

import asyncio
import itertools
import logging
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from findex.crawler.crawl import CrawlStats, Page, crawl
from findex.crawler.fetch import make_client
from findex.crawler.html import parse_html
from findex.crawler.urls import canonicalize, in_domains


# --- a fake website ----------------------------------------------------------------------
def html(title: str, *links: str, body: str = "", head: str = "") -> str:
    anchors = "".join(f'<a href="{href}">link</a>' for href in links)
    return f"<html><head><title>{title}</title>{head}</head><body>{body}{anchors}</body></html>"


@dataclass
class FakeWeb:
    """Serves `pages` ({url: html or (status, headers, body)}); everything else is 404.
    Records every request, when it started, and the most requests in flight at once."""

    pages: dict[str, str | tuple[int, dict[str, str], str]]
    latency: float = 0.0
    requests: list[str] = field(default_factory=list[str])
    starts: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    in_flight: Counter[str] = field(default_factory=Counter[str])
    max_in_flight: Counter[str] = field(default_factory=Counter[str])
    _cache: dict[str, bytes] = field(default_factory=dict[str, bytes])

    async def handler(self, request: httpx.Request) -> httpx.Response:
        url, host = str(request.url), request.url.host
        self.requests.append(url)
        self.starts.setdefault(host, []).append(time.perf_counter())
        self.in_flight[host] += 1
        self.in_flight["*"] += 1
        self.max_in_flight[host] = max(self.max_in_flight[host], self.in_flight[host])
        self.max_in_flight["*"] = max(self.max_in_flight["*"], self.in_flight["*"])
        try:
            await asyncio.sleep(self.latency)
            page = self.pages.get(url)
            if page is None:
                return httpx.Response(404, text="not found")
            if isinstance(page, str):
                return httpx.Response(200, headers=HTML_HEADERS, content=self._encoded(page))
            status, headers, body = page
            return httpx.Response(status, headers=headers, text=body)
        finally:
            self.in_flight[host] -= 1
            self.in_flight["*"] -= 1

    def _encoded(self, page: str) -> bytes:
        # Encode each page once, not per request: keeps the fake server's own CPU work
        # out of the event-loop timings the debug-mode test measures.
        if page not in self._cache:
            self._cache[page] = page.encode()
        return self._cache[page]

    def fetched(self) -> Counter[str]:
        return Counter(self.requests)


# One process pool for page parsing, shared by every crawl in this file: starting worker
# processes ("spawn") takes a moment, and the tests run dozens of crawls.
POOL: ProcessPoolExecutor


@pytest.fixture(scope="module", autouse=True)
def parse_pool() -> Iterator[None]:
    global POOL
    POOL = ProcessPoolExecutor(2, mp_context=mp.get_context("spawn"))
    yield
    POOL.shutdown()


def run_crawl(web: FakeWeb, seeds: list[str], **options: object) -> list[Page]:
    async def main() -> list[Page]:
        async with make_client(httpx.MockTransport(web.handler)) as client:
            pages = crawl(seeds, client=client, executor=POOL, min_delay=0.0, **options)
            async with aclosing(pages):
                return [p async for p in pages]

    return asyncio.run(main())


HTML_HEADERS = {"Content-Type": "text/html; charset=utf-8"}
SITE = "https://site.org"
SMALL_SITE: dict[str, str | tuple[int, dict[str, str], str]] = {
    f"{SITE}/": html("Home", "/a", "/b", "https://other.org/x", "mailto:me@site.org"),
    f"{SITE}/a": html("A", "/", "/b", "/a#section", "/c?y=2&x=1"),
    f"{SITE}/b": html("B", "/a/", "/c?x=1&y=2", "/photo.jpg"),
    f"{SITE}/c?x=1&y=2": html("C", "/"),
    "https://other.org/x": html("Other site"),
}


# --- the crawl -----------------------------------------------------------------------------
def test_crawls_the_whole_site_once_per_page() -> None:
    web = FakeWeb(SMALL_SITE)
    pages = run_crawl(web, [f"{SITE}/"])
    assert sorted(p.title for p in pages) == ["A", "B", "C", "Home"]
    # /a#section, /a/ and /c?y=2&x=1 are spellings of pages already seen: fetched once each
    assert max(web.fetched().values()) == 1
    assert "https://other.org/x" not in web.requests  # not in the allowed domains
    assert all(not u.endswith(".jpg") for u in web.requests)  # never even requested


def test_page_records() -> None:
    page = run_crawl(FakeWeb({f"{SITE}/": html("Home", body="<p>Hello <b>world</b></p>")}), [SITE])[
        0
    ]
    assert (page.url, page.title, page.text) == (f"{SITE}/", "Home", "Hello world")
    assert page.fetched_at.tzinfo is not None


def test_stops_at_max_pages() -> None:
    # A site of 100 pages, each linking to the next ten.
    pages_ = {
        f"{SITE}/p{i}": html(f"P{i}", *(f"/p{j}" for j in range(i + 1, i + 11))) for i in range(100)
    }
    web = FakeWeb(pages_)
    pages = run_crawl(web, [f"{SITE}/p0"], max_pages=7, concurrency=3)
    assert len(pages) == 7
    # it stopped promptly: no more than a few fetches beyond the budget (plus robots.txt)
    assert len(web.requests) <= 7 + 3 + 1


def test_robots_txt_is_obeyed() -> None:
    site = dict(SMALL_SITE)
    site[f"{SITE}/robots.txt"] = "User-agent: *\nDisallow: /b\n"
    web = FakeWeb(site)
    titles = {p.title for p in run_crawl(web, [f"{SITE}/"])}
    assert "B" not in titles and f"{SITE}/b" not in web.requests


def test_allowed_domains_include_subdomains() -> None:
    assert in_domains("https://docs.site.org/x", frozenset({"site.org"}))
    assert not in_domains("https://notsite.org/x", frozenset({"site.org"}))
    web = FakeWeb(
        {f"{SITE}/": html("Home", "https://docs.site.org/"), "https://docs.site.org/": html("Docs")}
    )
    assert {p.title for p in run_crawl(web, [SITE])} == {"Home", "Docs"}


# --- redirects -------------------------------------------------------------------------------
def test_redirect_targets_are_crawled_like_links() -> None:
    web = FakeWeb(
        {
            f"{SITE}/": html("Home", "/old"),
            f"{SITE}/old": (301, {"Location": "/new"}, ""),
            f"{SITE}/new": html("New"),
        }
    )
    assert {p.title for p in run_crawl(web, [SITE])} == {"Home", "New"}


def test_redirect_loop_ends() -> None:
    web = FakeWeb(
        {
            f"{SITE}/": html("Home", "/loop1"),
            f"{SITE}/loop1": (302, {"Location": "/loop2"}, ""),
            f"{SITE}/loop2": (302, {"Location": "/loop1"}, ""),
        }
    )
    assert [p.title for p in run_crawl(web, [SITE])] == ["Home"]
    assert web.fetched()[f"{SITE}/loop1"] == 1 and web.fetched()[f"{SITE}/loop2"] == 1


def test_trailing_slash_redirect_is_followed() -> None:
    # /docs and /docs/ have the same canonical form, so the frontier can't queue the target.
    web = FakeWeb(
        {
            f"{SITE}/": html("Home", "/docs"),
            f"{SITE}/docs": (301, {"Location": f"{SITE}/docs/"}, ""),
            f"{SITE}/docs/": html("Docs"),
        }
    )
    assert {p.title for p in run_crawl(web, [SITE])} == {"Home", "Docs"}


# --- what is and isn't kept --------------------------------------------------------------------
def test_non_html_noindex_and_nofollow() -> None:
    web = FakeWeb(
        {
            f"{SITE}/": html("Home", "/data", "/hidden", "/nofollow"),
            f"{SITE}/data": (200, {"Content-Type": "application/json"}, "{}"),
            f"{SITE}/hidden": html(
                "Hidden", "/found-via-hidden", head='<meta name="robots" content="noindex">'
            ),
            f"{SITE}/found-via-hidden": html("Found"),
            f"{SITE}/nofollow": html(
                "NoFollow", "/never", head='<meta name="robots" content="nofollow">'
            ),
            f"{SITE}/never": html("Never"),
        }
    )
    stats = CrawlStats()
    titles = {p.title for p in run_crawl(web, [SITE], stats=stats)}
    assert titles == {"Home", "Found", "NoFollow"}  # noindex: not kept, but its links followed
    assert f"{SITE}/never" not in web.requests
    assert stats.skipped == 2  # the JSON and the noindex page


def test_errors_are_counted_and_the_crawl_continues() -> None:
    web = FakeWeb({f"{SITE}/": html("Home", "/missing", "/ok"), f"{SITE}/ok": html("OK")})
    stats = CrawlStats()
    assert {p.title for p in run_crawl(web, [SITE], stats=stats)} == {"Home", "OK"}
    assert stats.errors == 1 and stats.pages == 2  # /missing -> 404


# --- politeness: concurrency and delays ----------------------------------------------------------
def test_per_host_and_global_concurrency_limits() -> None:
    pages_ = {f"https://h{h}.org/": html(f"H{h}", *(f"/p{i}" for i in range(8))) for h in range(3)}
    pages_ |= {f"https://h{h}.org/p{i}": html(f"H{h}P{i}") for h in range(3) for i in range(8)}
    web = FakeWeb(pages_, latency=0.02)
    hosts = [f"https://h{h}.org/" for h in range(3)]
    pages = run_crawl(
        web, hosts, concurrency=4, per_host=2, allowed_domains=["h0.org", "h1.org", "h2.org"]
    )
    assert len(pages) == 27
    assert all(web.max_in_flight[f"h{h}.org"] <= 2 for h in range(3))
    assert web.max_in_flight["*"] <= 4
    assert web.max_in_flight["*"] >= 3  # and it really did run requests concurrently


def test_minimum_delay_between_requests_to_a_host() -> None:
    web = FakeWeb(
        {f"{SITE}/": html("Home", *(f"/p{i}" for i in range(5)))}
        | {f"{SITE}/p{i}": html(f"P{i}") for i in range(5)}
    )

    async def main() -> None:
        async with (
            make_client(httpx.MockTransport(web.handler)) as client,
            aclosing(
                crawl([SITE], client=client, executor=POOL, min_delay=0.1, concurrency=5)
            ) as pages,
        ):
            async for _ in pages:
                pass

    asyncio.run(main())
    starts = web.starts["site.org"][1:]  # [0] is robots.txt, fetched before any page
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    # Margin: on Windows the clock ticks every ~15 ms, and asyncio may run a timer up to
    # one tick early, so allow 0.02 s below the 0.1 s delay.
    assert len(gaps) == 5 and min(gaps) >= 0.08


def test_crawl_delay_from_robots_txt_is_used() -> None:
    web = FakeWeb(
        {
            f"{SITE}/robots.txt": "User-agent: *\nCrawl-delay: 0.2\n",  # a fraction, on purpose
            f"{SITE}/": html("Home", "/a"),
            f"{SITE}/a": html("A"),
        }
    )
    run_crawl(web, [SITE], concurrency=5)
    starts = web.starts["site.org"][1:]
    assert starts[1] - starts[0] >= 0.18  # 0.2 s, minus a Windows clock tick of margin


# --- shutdown and errors ------------------------------------------------------------------------
def test_stopping_early_leaves_no_tasks_behind() -> None:
    web = FakeWeb({f"{SITE}/p{i}": html(f"P{i}", f"/p{i + 1}", f"/p{i + 2}") for i in range(50)})

    async def main() -> tuple[int, int]:
        async with make_client(httpx.MockTransport(web.handler)) as client:
            async with aclosing(
                crawl([f"{SITE}/p0"], client=client, executor=POOL, min_delay=0, concurrency=5)
            ) as pages:
                taken = 0
                async for _ in pages:
                    taken += 1
                    if taken == 2:
                        break  # the caller stops early
            others = asyncio.all_tasks() - {asyncio.current_task()}
            assert not client.is_closed  # we passed it in, so the crawler must not close it
            return taken, len(others)

    assert asyncio.run(main()) == (2, 0)  # every worker and the runner are gone


def test_an_unexpected_error_inside_the_crawl_is_raised() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("bug in the server simulation")

    async def main() -> None:
        async with (
            make_client(httpx.MockTransport(broken)) as client,
            aclosing(crawl([SITE], client=client, executor=POOL, min_delay=0)) as pages,
        ):
            async for _ in pages:
                pass

    with pytest.raises(ExceptionGroup) as info:
        asyncio.run(main())
    assert info.group_contains(RuntimeError, match="bug in the server simulation")


def test_parsing_never_blocks_the_event_loop(caplog: pytest.LogCaptureFixture) -> None:
    """asyncio's debug mode logs every step that holds the event loop longer than
    slow_callback_duration. That threshold is set to HALF the time parsing this page
    takes, measured here first: if parsing ran on the loop, its step would be flagged.
    Comparing against the machine's own parse time keeps the test meaningful anywhere --
    a slow machine, coverage or a debugger slow both sides alike, where a fixed 0.1 s
    threshold failed on a slower Windows run."""
    big = html("Big", *(f"/p{i}" for i in range(300)), body="<p>word </p>" * 40_000)
    t0 = time.perf_counter()
    parse_html(big, SITE)
    parse_time = time.perf_counter() - t0
    web = FakeWeb({f"{SITE}/": big})

    async def main() -> int:
        asyncio.get_running_loop().slow_callback_duration = parse_time / 2
        async with (
            make_client(httpx.MockTransport(web.handler)) as client,
            aclosing(
                crawl([SITE], client=client, executor=POOL, min_delay=0, max_pages=1)
            ) as pages,
        ):
            return len([p async for p in pages])

    with caplog.at_level(logging.WARNING, logger="asyncio"):
        assert asyncio.run(main(), debug=True) == 1
    slow = [r.getMessage() for r in caplog.records if "Executing" in r.getMessage()]
    assert slow == [], (
        f"parse takes {parse_time:.3f}s; a loop step took over half of that:\n" + "\n".join(slow)
    )


def test_seeds_must_be_web_urls() -> None:
    with pytest.raises(ValueError, match="seed"):
        run_crawl(FakeWeb({}), ["mailto:x@y.org", "not a url"])


# --- canonicalization and parsing --------------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTP://Example.ORG/", "http://example.org/"),
        ("https://example.org", "https://example.org/"),
        ("https://example.org:443/a", "https://example.org/a"),
        ("http://example.org:8080/a", "http://example.org:8080/a"),
        ("https://example.org/a/", "https://example.org/a"),
        ("https://example.org/a#frag", "https://example.org/a"),
        ("https://example.org/a?b=2&a=1", "https://example.org/a?a=1&b=2"),
        ("https://example.org/a?", "https://example.org/a"),
        ("https://example.org/a?q=x%20y", "https://example.org/a?q=x%20y"),
        ("https://example.org/report.PDF", None),
        ("mailto:me@example.org", None),
        ("javascript:void(0)", None),
        ("ftp://example.org/file", None),
        ("http://[broken", None),
    ],
)
def test_canonicalize(url: str, expected: str | None) -> None:
    assert canonicalize(url) == expected


def test_canonicalize_resolves_relative_links() -> None:
    assert canonicalize("../b/", "https://x.org/docs/a/page") == "https://x.org/docs/b"


def test_parse_html() -> None:
    page = parse_html(
        "<html><head><title> My  Page </title><style>p{}</style><base href='https://cdn.x.org/'>"
        "</head><body><h1>Hi &amp; bye</h1><script>ignored()</script>"
        "<a href='a'>A</a><a href='b' rel='nofollow'>B</a></body></html>",
        "https://x.org/dir/index.html",
    )
    assert page.title == "My Page"
    assert page.text == "Hi & bye A B"  # no script/style; the nofollow link's text stays
    assert page.links == ["https://cdn.x.org/a"]  # <base> applies; rel=nofollow skipped


def test_a_crawl_starts_and_stops_its_own_parsing_processes() -> None:
    web = FakeWeb({f"{SITE}/": html("Home", "/a"), f"{SITE}/a": html("A")})

    async def main() -> list[str]:
        async with (
            make_client(httpx.MockTransport(web.handler)) as client,
            aclosing(crawl([SITE], client=client, min_delay=0)) as pages,  # no executor given
        ):
            return [p.title async for p in pages]

    before = set(mp.active_children())  # e.g. the shared pool's workers, if started
    assert sorted(asyncio.run(main())) == ["A", "Home"]
    assert set(mp.active_children()) == before  # the crawl's own pool has shut down


# --- findex crawl (the command) ------------------------------------------------------------
@pytest.fixture
def fake_site(monkeypatch: pytest.MonkeyPatch) -> FakeWeb:
    """The `findex crawl` command builds its own HTTP client: make it talk to a fake site."""
    import findex.crawler.crawl

    web = FakeWeb(
        {
            f"{SITE}/": html("Home", "/event-loop", "/gone", "/private/x", body="<p>Welcome.</p>"),
            f"{SITE}/event-loop": html("Event loop", body="<p>The event loop runs coroutines.</p>"),
            f"{SITE}/private/x": html("Secret"),
            f"{SITE}/robots.txt": "User-agent: *\nDisallow: /private/\n",
        }
    )
    monkeypatch.setattr(
        findex.crawler.crawl, "make_client", lambda: make_client(httpx.MockTransport(web.handler))
    )
    return web


def test_crawl_command_end_to_end(fake_site: FakeWeb, tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from findex.cli import app

    runner = CliRunner()
    out = tmp_path / "crawl.jsonl"
    result = runner.invoke(app, ["crawl", SITE, "--out", str(out), "--delay", "0"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # everything goes to stderr: progress, summary, warnings
    assert "crawled 2 pages" in result.stderr
    assert "404" not in result.stderr  # an everyday 404 goes to the log file, not the screen

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["title"] for r in records) == ["Event loop", "Home"]
    assert set(records[0]) == {"id", "url", "title", "text", "fetched_at"}

    log = (tmp_path / "crawl.log").read_text(encoding="utf-8")
    assert f"200 {SITE}/event-loop" in log  # every URL's outcome is logged
    assert f"404 {SITE}/gone" in log
    assert f"robots.txt disallows {SITE}/private/x" in log
    assert f"{SITE}/private/x" not in fake_site.requests  # and robots.txt was obeyed

    # The output is a corpus: findex index reads it, findex search finds the pages.
    index = tmp_path / "crawl.bin"
    assert runner.invoke(app, ["index", str(out), "-o", str(index)]).exit_code == 0
    found = runner.invoke(app, ["search", str(index), "coroutines", "--json"])
    assert [json.loads(line)["title"] for line in found.stdout.splitlines()] == ["Event loop"]


def test_crawl_command_max_pages(fake_site: FakeWeb, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from findex.cli import app

    out = tmp_path / "c.jsonl"
    result = CliRunner().invoke(
        app, ["crawl", SITE, "-o", str(out), "--delay", "0", "--max-pages", "1"]
    )
    assert result.exit_code == 0 and len(out.read_text().splitlines()) == 1


def test_crawl_command_errors(fake_site: FakeWeb, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from findex.cli import app

    runner = CliRunner()
    bad = runner.invoke(app, ["crawl", "not-a-url", "-o", str(tmp_path / "x.jsonl")])
    assert bad.exit_code == 2 and "http(s) URLs" in bad.stderr
    fake_site.pages[f"{SITE}/robots.txt"] = "User-agent: *\nDisallow: /\n"
    blocked = runner.invoke(app, ["crawl", SITE, "-o", str(tmp_path / "y.jsonl"), "--delay", "0"])
    assert blocked.exit_code == 1 and "no pages crawled" in blocked.stderr
