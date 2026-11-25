"""In-memory inverted index, built in a single streaming pass.

    python -m findex.index data/simplewiki.jsonl [--limit N] [--term cat]
    python -m findex.index data/simplewiki.jsonl --out index.bin [--format binary|pickle]
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .corpus import Document, iter_documents
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int  # internal sequential id: 0, 1, 2, ...
    tf: int      # how many times the term occurs in that document


@dataclass(frozen=True, slots=True)
class DocMeta:
    ext_id: str  # the id from the corpus (e.g. the Wikipedia article id)
    path: str
    title: str


@dataclass(slots=True)
class Index:
    postings: dict[str, list[Posting]]  # term -> postings, sorted by doc_id
    doc_lengths: dict[int, int]         # doc_id -> number of tokens (BM25 needs it in Lab 3)
    doc_meta: dict[int, DocMeta]        # doc_id -> info for display

    @property
    def n_docs(self) -> int:
        return len(self.doc_meta)

    def get(self, term: str) -> list[Posting]:
        """Postings for a term, or [] if unknown. Never inserts anything."""
        return self.postings.get(term, [])


def build_index(docs: Iterable[Document]) -> Index:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):  # ascending ids => every list stays sorted for free
        counts = Counter(tokenize(doc.text))  # this document's term frequencies, then discarded
        doc_lengths[doc_id] = sum(counts.values())
        doc_meta[doc_id] = DocMeta(ext_id=doc.doc_id, path=str(doc.path), title=doc.title)
        for term, tf in counts.items():
            postings[term].append(Posting(doc_id, tf))

    # Convert to a plain dict: a defaultdict would silently create an empty entry
    # for every unknown term that someone looks up.
    return Index(dict(postings), doc_lengths, doc_meta)


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.index")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--term", default=None, help="print the postings for this term")
    ap.add_argument("--out", type=Path, default=None, help="save the index to this file")
    ap.add_argument("--format", choices=("binary", "pickle"), default="binary")
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(args.root), args.limit))
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_postings = sum(len(p) for p in index.postings.values())
    print(f"documents:    {index.n_docs}")
    print(f"terms:        {len(index.postings)}")
    print(f"postings:     {n_postings}")
    print(f"elapsed:      {elapsed:.2f}s")
    print(f"peak memory:  {peak / 1e6:.1f} MB")
    if args.out:
        from .store import save  # imported here: store itself imports this module

        t0 = time.perf_counter()
        save(index, args.out, args.format)
        print(f"saved:        {args.out} ({args.format}, "
              f"{args.out.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.2f}s)")
    if args.term:
        plist = index.get(args.term)
        print(f"'{args.term}': in {len(plist)} documents; first 5:")
        for p in plist[:5]:
            print(f"  doc {p.doc_id} ({index.doc_meta[p.doc_id].title!r}) tf={p.tf}")


if __name__ == "__main__":
    # When run with `python -m findex.index`, this file is the module "__main__", so the
    # classes defined above would be pickled as "__main__.Posting" and could not be loaded
    # from anywhere else. Re-importing under the real name avoids that.
    from findex.index import main as _main

    _main()