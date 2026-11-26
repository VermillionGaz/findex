"""robots.txt, fetched once per site and cached for the rest of the crawl.

    robots = RobotsCache(client)
    if await robots.allowed("https://example.org/page"):
        ...

Rules are read per RFC 9309 by RobotsRules below (not urllib.robotparser: see there).

What a robots.txt fetch's outcome means (RFC 9309, plus one cautious choice):
  2xx                 -> parse and obey the rules
  401 / 403           -> treat the site as off-limits (cautious, like Python's robotparser)
  any other 4xx       -> no rules: everything allowed (404 is the common case)
  5xx, timeout, ...   -> the site is unreachable: treat everything as disallowed
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from . import USER_AGENT
from .fetch import Sleep, fetch

log = logging.getLogger("findex.crawler.robots")


@dataclass(frozen=True, slots=True)
class _Rule:
    allow: bool
    pattern: str  # as written, e.g. "/*?wprov="; its length decides precedence
    regex: re.Pattern[str]


@dataclass(slots=True)
class _Group:
    agents: list[str] = field(default_factory=list[str])  # lowercased User-agent values
    rules: list[_Rule] = field(default_factory=list[_Rule])
    delay: float | None = None


class RobotsRules:
    """robots.txt rules, per RFC 9309 -- written here because Python's urllib.robotparser
    falls short of the standard in ways real sites hit:

      * wildcards: "Disallow: /*?wprov=" uses "*" (any characters) and rules may end in
        "$" (end of URL); robotparser compares paths literally, so such rules never match;
      * precedence: the most specific (longest) matching rule wins, and Allow wins a tie;
        robotparser takes the first rule that matches, in file order;
      * Crawl-delay with a fraction ("0.5"), which robotparser ignores.

    A group applies to us if one of its User-agent values equals our product token
    ("findex" in "findex/0.4.0 (+https://...)"), case-insensitively; groups naming us
    are combined. If none names us, the "*" groups apply.
    """

    def __init__(self, groups: list[_Group], *, allow_all: bool = True) -> None:
        self._groups = groups
        self._allow_all = allow_all  # the answer when no rule matches

    @classmethod
    def parse(cls, text: str) -> RobotsRules:
        groups: list[_Group] = []
        current: _Group | None = None
        in_rules = False  # a User-agent line after rules starts a new group
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if current is None or in_rules:
                    current = _Group()
                    groups.append(current)
                    in_rules = False
                current.agents.append(value.lower())
            elif current is not None and key in ("allow", "disallow"):
                in_rules = True
                if value:  # an empty "Disallow:" means "nothing is disallowed"
                    current.rules.append(_Rule(key == "allow", value, _compile(value)))
            elif current is not None and key == "crawl-delay":
                in_rules = True
                try:
                    current.delay = float(value)
                except ValueError:
                    pass
            # Sitemap and unknown keys don't belong to a group; lines before any
            # User-agent line belong to no group and are ignored, as the RFC says.
        return cls(groups)

    @classmethod
    def everything(cls, allowed: bool) -> RobotsRules:
        return cls([], allow_all=allowed)

    def _groups_for(self, user_agent: str) -> list[_Group]:
        token = user_agent.split("/")[0].strip().lower()
        ours = [g for g in self._groups if token in g.agents]
        return ours or [g for g in self._groups if "*" in g.agents]

    def allowed(self, user_agent: str, url: str) -> bool:
        if not self._allow_all:
            return False
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        if path == "/robots.txt":
            return True
        best: _Rule | None = None
        for group in self._groups_for(user_agent):
            for rule in group.rules:
                if rule.regex.match(path) and (
                    best is None
                    or len(rule.pattern) > len(best.pattern)
                    or (len(rule.pattern) == len(best.pattern) and rule.allow)
                ):
                    best = rule
        return best is None or best.allow

    def crawl_delay(self, user_agent: str) -> float | None:
        delays = [g.delay for g in self._groups_for(user_agent) if g.delay is not None]
        return max(delays) if delays else None


def _compile(pattern: str) -> re.Pattern[str]:
    """A robots.txt path pattern as a regex anchored at the start: "*" matches any
    characters, a final "$" anchors the end; everything else is literal."""
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + ("$" if anchored else ""))


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
        self._rules: dict[str, RobotsRules] = {}
        # One lock per site: when 20 workers ask about the same new site at once, one of
        # them fetches robots.txt and the others wait for it, instead of 20 fetches.
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        return (await self._get(url)).allowed(self._user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        """The site's Crawl-delay for us, in seconds, if it sets one."""
        return (await self._get(url)).crawl_delay(self._user_agent)

    async def _get(self, url: str) -> RobotsRules:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        if origin in self._rules:  # fast path: no lock needed once cached
            return self._rules[origin]
        async with self._locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._rules:  # someone else may have fetched it while we waited
                self._rules[origin] = await self._fetch_rules(origin)
            return self._rules[origin]

    async def _fetch_rules(self, origin: str) -> RobotsRules:
        result = await fetch(
            self._client,
            f"{origin}/robots.txt",
            follow_redirects=True,
            max_bytes=500_000,
            sleep=self._sleep,
        )
        if result.ok:
            rules = RobotsRules.parse(result.content.decode("utf-8", errors="replace"))
            log.info("%s/robots.txt: %d bytes of rules", origin, len(result.content))
            return rules
        if result.status in (401, 403):
            log.warning(
                "%s/robots.txt: HTTP %d, treating the site as off-limits", origin, result.status
            )
            return RobotsRules.everything(allowed=False)
        if result.status is not None and 400 <= result.status < 500:
            log.info("%s/robots.txt: HTTP %d, no rules", origin, result.status)
            return RobotsRules.everything(allowed=True)
        log.warning(
            "%s/robots.txt unreachable (%s): treating the site as off-limits", origin, result.error
        )
        return RobotsRules.everything(allowed=False)
