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


# --- RFC 9309 rules (RobotsRules), where Python's robotparser falls short ----------------
from findex.crawler.robots import RobotsRules

UA = "findex/0.4.0 (+https://github.com/VermillionGaz/findex)"


def allowed(robots_txt: str, path: str, user_agent: str = UA) -> bool:
    return RobotsRules.parse(robots_txt).allowed(user_agent, f"https://x.org{path}")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/wiki/Page", True),
        ("/wiki/Page?wprov=sfla1", False),  # "*" matches "wiki/Page"
        ("/?wprov=x", False),
        ("/wiki/Page?other=1", True),
    ],
)
def test_wildcards(path: str, expected: bool) -> None:
    assert allowed("User-agent: *\nDisallow: /*?wprov=\n", path) is expected


def test_dollar_anchors_the_end() -> None:
    rules = "User-agent: *\nDisallow: /*.php$\n"
    assert not allowed(rules, "/index.php")
    assert allowed(rules, "/index.php?title=X")  # doesn't end in .php
    assert allowed(rules, "/php-guide")


def test_longest_match_wins_not_first_match() -> None:
    # robotparser would take the first rule (Disallow) and refuse /wiki/Allowed_page.
    rules = "User-agent: *\nDisallow: /wiki/\nAllow: /wiki/Allowed\n"
    assert allowed(rules, "/wiki/Allowed_page")
    assert not allowed(rules, "/wiki/Other")


def test_allow_wins_a_tie() -> None:
    assert allowed("User-agent: *\nDisallow: /page\nAllow: /page\n", "/page")


def test_groups_naming_us_are_combined_and_override_star() -> None:
    rules = (
        "User-agent: *\nDisallow: /\n\n"
        "User-agent: findex\nDisallow: /a\n\n"
        "User-agent: FINDEX\nDisallow: /b\n"  # matching is case-insensitive
    )
    assert allowed(rules, "/c")  # the * group doesn't apply to us at all
    assert not allowed(rules, "/a") and not allowed(rules, "/b")


def test_agent_names_match_the_product_token_exactly() -> None:
    rules = "User-agent: find\nDisallow: /\n"  # a different robot, merely a prefix of ours
    assert allowed(rules, "/page")


def test_empty_disallow_and_robots_txt_itself() -> None:
    assert allowed("User-agent: *\nDisallow:\n", "/anything")
    assert allowed("User-agent: *\nDisallow: /\n", "/robots.txt")


def test_rules_before_any_user_agent_line_are_ignored() -> None:
    assert allowed("Disallow: /\nUser-agent: *\nDisallow: /private\n", "/public")


def test_crawl_delays_including_fractions() -> None:
    text = """
    User-agent: *
    Crawl-delay: 0.5          # a fraction: Python's robotparser would ignore this line

    User-agent: findex
    User-agent: otherbot
    Disallow: /x
    Crawl-delay: 3

    User-agent: badbot
    Crawl-delay: soon
    """
    rules = RobotsRules.parse(text)
    assert rules.crawl_delay(UA) == 3.0
    assert rules.crawl_delay("otherbot") == 3.0
    assert rules.crawl_delay("somebot/1.0") == 0.5
    # badbot has its own group, so the * group doesn't apply to it; its own value is
    # unparsable, so it has no delay at all.
    assert rules.crawl_delay("badbot") is None


LIMBUS_WIKI = """
User-agent: *
# WMF Provenance params
Disallow: /*?wprov=

## SEO and intelligence bots
User-agent: AhrefsBot
Disallow: /

User-agent: ClaudeBot
Disallow: /

# wget in recursive mode is a bit disastrous when --wait is not used - sorry!
User-agent: wget
Disallow: /

User-agent: SiteSucker
Crawl-delay: 1

Sitemap: https://limbuscompany.wiki.gg/sitemaps/sitemap-index-limbuscompany_en.xml
"""


def test_the_limbus_wiki_robots_txt() -> None:
    """The rules from limbuscompany.wiki.gg (with the User-agent: * line their first rule needs)."""
    rules = RobotsRules.parse(LIMBUS_WIKI)
    base = "https://limbuscompany.wiki.gg"
    assert rules.allowed(UA, f"{base}/wiki/Faust")
    assert not rules.allowed(UA, f"{base}/wiki/Faust?wprov=sfti1")  # robotparser: allowed!
    assert not rules.allowed("ClaudeBot/1.0", f"{base}/wiki/Faust")
    assert not rules.allowed("Wget/1.21", f"{base}/wiki/Faust")
    assert rules.crawl_delay(UA) is None  # only SiteSucker is given a delay
