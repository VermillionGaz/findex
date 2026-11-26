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


def normalize(url: str, base: str | None = None) -> str | None:
    """The URL to request: url resolved against base, scheme and host lowercased, default
    port and fragment dropped, query parameters sorted -- but the path kept as written,
    trailing slash included. None if it's not worth fetching: not http(s), no host, or a
    file type we can't index.

    The path is kept because robots.txt rules match it literally: a page that links to
    "/private/" must be checked as "/private/", or the rule "Disallow: /private/" would
    not match a slash-stripped "/private".
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
    if any(path.lower().rstrip("/").endswith(ext) for ext in SKIP_EXTENSIONS):
        return None
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)), quote_via=quote)
    return urlunsplit((scheme, netloc, path, query, ""))


def canonicalize(url: str, base: str | None = None) -> str | None:
    """The page's identity, for the `seen` set: normalize() plus trailing slash removed
    (except the root "/"), since "/docs" and "/docs/" are the same page.

    canonicalize("HTTP://Example.org:80/docs/?b=2&a=1#intro") -> "http://example.org/docs?a=1&b=2"
    """
    normal = normalize(url, base)
    if normal is None:
        return None
    parts = urlsplit(normal)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def in_domains(url: str, domains: frozenset[str]) -> bool:
    """Is url's host one of domains, or a subdomain of one? ("docs.x.org" is in "x.org")"""
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)
