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