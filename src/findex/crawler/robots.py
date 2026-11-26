"""robots.txt, fetched once per site and cached for the rest of the crawl.

    robots = RobotsCache(client)
    if await robots.allowed("https://example.org/page"):
        ...

What a robots.txt fetch's outcome means (RFC 9309, plus one cautious choice):
  2xx                 -> parse and obey the rules
  401 / 403           -> treat the site as off-limits (cautious, like Python's robotparser)
  any other 4xx       -> no rules: everything allowed (404 is the common case)
  5xx, timeout, ...   -> the site is unreachable: treat everything as disallowed
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from . import USER_AGENT
from .fetch import Sleep, fetch

log = logging.getLogger("findex.crawler.robots")


@dataclass(frozen=True, slots=True)
class _Rules:
    parser: RobotFileParser | None  # None: no rules to consult
    allow_all: bool  # used only when parser is None
    delays: dict[str, float] = field(default_factory=dict[str, float])  # agent -> Crawl-delay

    def allowed(self, user_agent: str, url: str) -> bool:
        return self.parser.can_fetch(user_agent, url) if self.parser else self.allow_all

    def crawl_delay(self, user_agent: str) -> float | None:
        # Same matching as robotparser: a group applies if its agent name appears in our
        # product token ("findex" in "findex/0.4.0"); otherwise the "*" group.
        token = user_agent.split("/")[0].lower()
        for agent, delay in self.delays.items():
            if agent != "*" and agent in token:
                return delay
        return self.delays.get("*")


def parse_crawl_delays(lines: list[str]) -> dict[str, float]:
    """Crawl-delay per user-agent group. Read here rather than by robotparser, which
    ignores any value that isn't a whole number -- "Crawl-delay: 0.5" would be lost."""
    delays: dict[str, float] = {}
    agents: list[str] = []
    in_rules = False  # a new User-agent line after rules starts a new group
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if in_rules:
                agents, in_rules = [], False
            agents.append(value.lower())
        else:
            in_rules = True
            if key == "crawl-delay":
                try:
                    delay = float(value)
                except ValueError:
                    continue
                for agent in agents:
                    delays.setdefault(agent, delay)
    return delays


class RobotsCache:
    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str = USER_AGENT,
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._sleep = sleep  # for fetch()'s retry waits; tests pass a fake
        self._rules: dict[str, _Rules] = {}
        # One lock per site: when 20 workers ask about the same new site at once, one of
        # them fetches robots.txt and the others wait for it, instead of 20 fetches.
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        return (await self._get(url)).allowed(self._user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        """The site's Crawl-delay for us, in seconds, if it sets one."""
        return (await self._get(url)).crawl_delay(self._user_agent)

    async def _get(self, url: str) -> _Rules:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        if origin in self._rules:  # fast path: no lock needed once cached
            return self._rules[origin]
        async with self._locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._rules:  # someone else may have fetched it while we waited
                self._rules[origin] = await self._fetch_rules(origin)
            return self._rules[origin]

    async def _fetch_rules(self, origin: str) -> _Rules:
        result = await fetch(
            self._client,
            f"{origin}/robots.txt",
            follow_redirects=True,
            max_bytes=500_000,
            sleep=self._sleep,
        )
        if result.ok:
            lines = result.content.decode("utf-8", errors="replace").splitlines()
            parser = RobotFileParser()
            # parse(), never read(): read() downloads with blocking urllib and would
            # freeze the event loop.
            parser.parse(lines)
            log.info("%s/robots.txt: %d bytes of rules", origin, len(result.content))
            return _Rules(parser, allow_all=False, delays=parse_crawl_delays(lines))
        if result.status in (401, 403):
            log.warning(
                "%s/robots.txt: HTTP %d, treating the site as off-limits", origin, result.status
            )
            return _Rules(None, allow_all=False)
        if result.status is not None and 400 <= result.status < 500:
            log.info("%s/robots.txt: HTTP %d, no rules", origin, result.status)
            return _Rules(None, allow_all=True)
        log.warning(
            "%s/robots.txt unreachable (%s): treating the site as off-limits", origin, result.error
        )
        return _Rules(None, allow_all=False)
