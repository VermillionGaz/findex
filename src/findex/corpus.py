"""Lazy document stream: one Document in memory at a time."""
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


def _iter_txt_dir(root: Path) -> Iterator[Document]:
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")  # strict: bad bytes raise
        except UnicodeDecodeError as exc:
            log.warning("skipping %s: %s", path, exc)
            continue
        yield Document(doc_id=str(path.relative_to(root)), path=path, text=text, title=path.stem)


def _iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8", errors="replace") as f:  # file = lazy line iterator
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield Document(doc_id=str(obj.get("id", n)), path=path, text=obj["text"], title=str(obj.get("title", "")))
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("skipping %s line %d: %s", path, n, exc)


def iter_documents(root: str | Path) -> Iterator[Document]:
    """Yield documents from a directory of .txt files or a .jsonl file."""
    root = Path(root)
    if not root.exists():
        # Without this check a typo in the path silently yields zero documents.
        # Note: because this is a generator, the error appears on the first next(),
        # not when iter_documents() is called.
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    if root.is_file():
        yield from _iter_jsonl(root)
    else:
        yield from _iter_txt_dir(root)