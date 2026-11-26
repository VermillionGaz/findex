"""findex's web crawler: polite, concurrent fetching with asyncio (Lab 6)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("findex")
    except PackageNotFoundError:  # running from a source checkout that isn't installed
        return "dev"


# Honest identification: what we are, and where a site owner can find out more or complain.
USER_AGENT = f"findex/{_version()} (+https://github.com/VermillionGaz/findex)"
