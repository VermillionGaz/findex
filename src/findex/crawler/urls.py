"""URL canonicalization: one spelling per page, so the crawler fetches it once.

canonicalize("HTTP://Example.org:80/docs/?b=2&a=1#intro")  ->  "http://example.org/docs?a=1&b=2"
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}

# Links to files a text search engine can't use: skipped without even asking for them.
SKIP_EXTENSIONS = frozenset(
    [
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dmg",
        ".iso",
        ".msi",
        ".apk",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".bmp",
        ".tif",
        ".tiff",
        ".avif",
        ".mp3",
        ".mp4",
        ".m4a",
        ".wav",
        ".ogg",
        ".webm",
        ".avi",
        ".mov",
        ".mkv",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".rss",
        ".atom",
        ".csv",
        ".xls",
        ".xlsx",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
    ]
)


def canonicalize(url: str, base: str | None = None) -> str | None:
    """The canonical form of url (resolved against base), or None if it's not worth
    fetching: not http(s), no host, or a file type we can't index.

    Scheme and host lowercased, default port dropped, fragment dropped (it's the same
    page), trailing slash removed (except the root "/"), query parameters sorted.
    """
    try:
        parts = urlsplit(urljoin(base, url) if base else url)
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # e.g. "http://[bad" or a port out of range
        return None
    if scheme not in DEFAULT_PORTS or not host:
        return None
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    if any(path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return None
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)), quote_via=quote)
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def in_domains(url: str, domains: frozenset[str]) -> bool:
    """Is url's host one of domains, or a subdomain of one? ("docs.x.org" is in "x.org")"""
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)
