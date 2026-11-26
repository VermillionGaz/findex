"""Snippets: a short window of a result's text with the query terms highlighted.

    "…the Sun and the **planets** of the **solar** **system** formed about 4.6…"

The index stores word positions but not the text itself, so fetch_texts()
re-reads it from the corpus: a .txt file directly, a .jsonl by streaming it up
to the highest wanted id. (Internal doc ids are positions in the corpus stream,
so the n-th document read IS doc n.)
"""
from __future__ import annotations

import unicodedata
from collections.abc import Collection, Iterable
from itertools import islice
from pathlib import Path

from .corpus import iter_documents
from .index import Index
from .tokenize import token_spans

RADIUS = 80                       # characters on each side of the best match
PLAIN = ("**", "**")              # Markdown bold: for files and pipes
ANSI = ("\x1b[1;33m", "\x1b[0m")  # bold yellow: for a terminal


def fetch_texts(index: Index, doc_ids: Iterable[int]) -> dict[int, str]:
    """Texts of the given documents, read back from the corpus. Documents whose
    source is missing or has changed since indexing are left out."""
    by_path: dict[str, list[int]] = {}
    for d in doc_ids:
        by_path.setdefault(index.doc_meta[d].path, []).append(d)

    texts: dict[int, str] = {}
    for path_str, ids in by_path.items():
        path = Path(path_str)
        if not path.is_file():
            continue
        if path.suffix == ".txt":                      # one document per file
            for d in ids:
                texts[d] = path.read_text(encoding="utf-8", errors="replace")
            continue
        wanted = set(ids)                              # a .jsonl: stream to the last one
        for n, doc in enumerate(islice(iter_documents(path), max(wanted) + 1)):
            if n in wanted and doc.doc_id == index.doc_meta[n].ext_id:  # same corpus?
                texts[n] = doc.text
    return texts


def make_snippet(
    text: str,
    terms: Collection[str],
    radius: int = RADIUS,
    mark: tuple[str, str] = PLAIN,
) -> str:
    """A ±radius-character window around the best match, terms wrapped in `mark`.

    "Best" = the window covering the most DIFFERENT query terms, then the most
    matches: for "solar system", a spot with both words beats one with "solar"
    three times.
    """
    text = unicodedata.normalize("NFC", text)          # token_spans' offsets are NFC
    wanted = set(terms)
    hits = [(tok, start, end) for tok, start, end in token_spans(text) if tok in wanted]
    if not hits:
        return _cut(text, 0, 2 * radius, [], mark)

    best, best_key = hits[0][1], (-1, -1)
    for _, center, _ in hits[:500]:                    # cap the work on huge documents
        inside = [tok for tok, start, _ in hits if abs(start - center) <= radius]
        key = (len(set(inside)), len(inside))          # distinct terms, then total matches
        if key > best_key:
            best, best_key = center, key
    return _cut(text, best - radius, best + radius, [(s, e) for _, s, e in hits], mark)


def _cut(text: str, lo: int, hi: int, hits: list[tuple[int, int]], mark: tuple[str, str]) -> str:
    """text[lo:hi], widened to whole words, with hits marked and … where text was cut."""
    lo, hi = max(lo, 0), min(hi, len(text))
    while lo > 0 and not text[lo - 1].isspace():       # don't start mid-word
        lo -= 1
    while hi < len(text) and not text[hi].isspace():   # don't end mid-word
        hi += 1
    out, cursor = [], lo
    for start, end in hits:
        if lo <= start and end <= hi:
            out += [text[cursor:start], mark[0], text[start:end], mark[1]]
            cursor = end
    out.append(text[cursor:hi])
    body = " ".join("".join(out).split())              # newlines/tabs -> single spaces
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(text) else "")
