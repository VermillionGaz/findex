"""Lazy document stream: one Document in memory at a time.

The corpus is read in *chunks*: a group of .txt files, or a byte range of a .jsonl
file that starts and ends on line boundaries. iter_documents() reads one chunk that
spans the whole corpus; parallel indexing (parallel.py) plans several chunks and
gives each to a worker. Both use read_chunk(), so they see exactly the same documents.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str
    title: str = ""


# --- chunks ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TxtChunk:
    """Some of the .txt files of a folder corpus, in sorted order."""

    root: Path  # doc ids are paths relative to this
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class JsonlChunk:
    """Lines [start, end) of a .jsonl file, as byte offsets that fall on line starts."""

    path: Path
    start: int
    end: int
    first_line: int  # 1-based number of the line at `start` (the default doc id)


# Small, frozen and made of paths and ints: cheap to pickle and send to a worker process.
type Chunk = TxtChunk | JsonlChunk


def read_chunk(chunk: Chunk) -> Iterator[Document]:
    """The documents of one chunk, in corpus order."""
    if isinstance(chunk, TxtChunk):
        yield from _read_txt(chunk)
    else:
        yield from _read_jsonl(chunk)


def _read_txt(chunk: TxtChunk) -> Iterator[Document]:
    for path in chunk.paths:
        try:
            text = path.read_text(encoding="utf-8")  # strict: bad bytes raise
        except UnicodeDecodeError as exc:
            log.warning("skipping %s: %s", path, exc)
            continue
        doc_id = str(path.relative_to(chunk.root))
        yield Document(doc_id=doc_id, path=path, text=text, title=path.stem)


def _read_jsonl(chunk: JsonlChunk) -> Iterator[Document]:
    # Binary mode, so byte offsets are exact; each line is decoded on its own.
    with chunk.path.open("rb") as f:
        f.seek(chunk.start)
        pos, n = chunk.start, chunk.first_line
        while pos < chunk.end:
            raw = f.readline()
            if not raw:
                break
            pos += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                doc = _parse_jsonl_line(line, chunk.path, n)
                if doc is not None:
                    yield doc
            n += 1


def _parse_jsonl_line(line: str, path: Path, n: int) -> Document | None:
    try:
        obj = json.loads(line)
        return Document(
            doc_id=str(obj.get("id", n)),
            path=path,
            text=obj["text"],
            title=str(obj.get("title", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        # TypeError/AttributeError: valid JSON that isn't an object, e.g. a bare number
        log.warning("skipping %s line %d: %s", path, n, exc)
        return None


# --- planning ------------------------------------------------------------------------------
def plan_chunks(root: str | Path, n_chunks: int = 1, limit: int | None = None) -> list[Chunk]:
    """Split a corpus into up to n_chunks contiguous chunks of roughly equal size.

    limit keeps only the first `limit` documents (for .jsonl: non-empty lines; a line
    that then fails to parse still counts toward the limit).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    if root.is_file():
        return _plan_jsonl(root, n_chunks, limit)
    paths = sorted(root.rglob("*.txt"))[:limit]
    size = max(1, -(-len(paths) // n_chunks))  # ceiling division: chunk sizes differ by <= 1
    return [TxtChunk(root, tuple(paths[i : i + size])) for i in range(0, len(paths), size)] or [
        TxtChunk(root, ())
    ]


def _plan_jsonl(path: Path, n_chunks: int, limit: int | None) -> list[Chunk]:
    end = path.stat().st_size if limit is None else _offset_after(path, limit)
    # Cut points near equal byte offsets, each moved forward to the next line start.
    cuts = [0]
    with path.open("rb") as f:
        for k in range(1, n_chunks):
            f.seek(end * k // n_chunks)
            f.readline()  # finish the line we landed in the middle of
            cut = min(f.tell(), end)
            if cut > cuts[-1]:
                cuts.append(cut)
    if cuts[-1] != end:
        cuts.append(end)
    # The line number at each cut: count newlines once, front to back.
    first_lines = _line_numbers_at(path, cuts[:-1])
    return [
        JsonlChunk(path, start, stop, line)
        for start, stop, line in zip(cuts[:-1], cuts[1:], first_lines, strict=True)
    ] or [JsonlChunk(path, 0, 0, 1)]


def _offset_after(path: Path, limit: int) -> int:
    """Byte offset just past the limit-th non-empty line."""
    seen = 0
    with path.open("rb") as f:
        pos = 0
        for raw in f:
            if seen == limit:
                break
            pos += len(raw)
            if raw.strip():
                seen += 1
    return pos


def _line_numbers_at(path: Path, offsets: list[int]) -> list[int]:
    """1-based line number of the line starting at each (ascending) byte offset."""
    numbers: list[int] = []
    line, pos = 1, 0
    with path.open("rb") as f:
        for target in offsets:
            while pos < target:  # count newlines between the previous offset and this one
                block = f.read(min(1 << 20, target - pos))
                line += block.count(b"\n")
                pos += len(block)
            numbers.append(line)
    return numbers


# --- the whole corpus, as one stream -------------------------------------------------------
def iter_documents(root: str | Path) -> Iterator[Document]:
    """Yield documents from a directory of .txt files or a .jsonl file."""
    root = Path(root)
    if not root.exists():
        # Without this check a typo in the path silently yields zero documents.
        # Note: because this is a generator, the error appears on the first next(),
        # not when iter_documents() is called.
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    if root.is_file():
        yield from read_chunk(JsonlChunk(root, 0, root.stat().st_size, 1))
    else:
        yield from read_chunk(TxtChunk(root, tuple(sorted(root.rglob("*.txt")))))
