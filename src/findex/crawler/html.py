"""Pull the title, visible text and links out of an HTML page (standard library only).

parse_html() is plain CPU work. The crawler runs it with asyncio.to_thread, so parsing
a big page doesn't stall the event loop while other requests are waiting to be handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

# Elements whose content is never visible text.
_INVISIBLE = frozenset({"script", "style", "noscript", "template", "svg", "head"})


@dataclass(slots=True)
class ParsedPage:
    title: str = ""
    text: str = ""
    links: list[str] = field(default_factory=list[str])  # absolute, in page order
    noindex: bool = False  # <meta name="robots" content="noindex">: don't keep this page
    nofollow: bool = False  # <meta name="robots" content="nofollow">: don't follow its links


class _Parser(HTMLParser):
    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)  # &amp; -> & etc. inside the text
        self.base = url
        self.page = ParsedPage()
        self._hidden = 0  # how deep we are inside invisible elements
        self._in_title = False
        self._title: list[str] = []
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "base" and a.get("href"):
            self.base = urljoin(self.base, a["href"])
        elif tag == "a" and a.get("href") and "nofollow" not in a.get("rel", "").lower().split():
            self.page.links.append(urljoin(self.base, a["href"].strip()))
        elif tag == "meta" and a.get("name", "").lower() == "robots":
            directives = {d.strip() for d in a.get("content", "").lower().split(",")}
            self.page.noindex |= bool(directives & {"noindex", "none"})
            self.page.nofollow |= bool(directives & {"nofollow", "none"})
        if tag == "title":
            self._in_title = True
        elif tag in _INVISIBLE:
            self._hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _INVISIBLE and self._hidden:
            self._hidden -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif not self._hidden:
            self._text.append(data)

    def result(self) -> ParsedPage:
        page = self.page
        page.title = " ".join("".join(self._title).split())  # collapse all whitespace
        page.text = " ".join(" ".join(self._text).split())
        if page.nofollow:
            page.links = []
        return page


def parse_html(html: str, url: str) -> ParsedPage:
    parser = _Parser(url)
    parser.feed(html)
    parser.close()
    return parser.result()
