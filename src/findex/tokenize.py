"""Streaming tokenizer.

Policy:
  * NFC-normalize, then casefold (so "Straße" == "STRASSE", "é" == "e\u0301").
  * Curly apostrophes (’ ‘ ʼ) are folded to a straight ' so "don't" == "don’t".
  * A token = letters/digits (Unicode-aware). Underscores are NOT part of tokens.
  * Internal apostrophes/hyphens keep a token together: don't, rock-n-roll.
    Leading/trailing ones are dropped: 'quoted' -> quoted.
  * Digits are kept, but "3.14" and "100,000" split at the punctuation.
  * Abbreviations split too: "U.S." -> u, s.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_TOKEN = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def tokenize(text: str) -> Iterator[str]:
    text = unicodedata.normalize("NFC", text).casefold().translate(_APOSTROPHES)
    for m in _TOKEN.finditer(text):  # finditer = lazy; findall would build a list
        yield m.group()

def token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """(token, start, end) for every token, with offsets into the NFC form of text.

    Same tokens as tokenize(), but it also says WHERE each one is, which snippets
    need for highlighting. The difference: here the regex runs before casefolding,
    so the offsets point into readable text rather than its casefolded copy (which
    can have a different length: "ß".casefold() == "ss"). Casefolding the match
    afterwards gives the same token in practice.
    """
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    for m in _TOKEN.finditer(text):
        yield m.group().casefold(), m.start(), m.end()
