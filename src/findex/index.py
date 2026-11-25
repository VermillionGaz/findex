"""In-memory inverted index, built in a single streaming pass.

    python -m findex.index data/simplewiki.jsonl [--limit N] [--term cat]
    python -m findex.index data/simplewiki.jsonl --out index.bin [--format binary|pickle]
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path

from .corpus import Document, iter_documents
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int  # internal sequential id: 0, 1, 2, ...
    tf: int      # how many times the term occurs in that document
    # Token offsets of each occurrence, ascending (len == tf). Empty if the index
    # was built without positions. Needed for phrase queries and snippets.
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    ext_id: str  # the id from the corpus (e.g. the Wikipedia article id)
    path: str
    title: str


class Index(Mapping[str, list[Posting]]):
    """An inverted index: term -> postings, sorted by doc_id.

    Behaves like a read-only dict of postings, because it IS a Mapping:
        len(ix)       number of distinct terms
        "cat" in ix   is the term indexed?
        ix["cat"]     its postings (KeyError if absent)
        for t in ix   every term
        ix.get("cat", [])   postings or a default -- inherited from Mapping

    Treat it as immutable once built: avg_doc_length is computed once and cached.
    """

    # No __slots__ here: cached_property stores its value in the instance __dict__.
    # Slots pay off for millions of Postings, not for one Index.

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self.postings = postings          # term -> postings, sorted by doc_id
        self.doc_lengths = doc_lengths    # doc_id -> number of tokens
        self.doc_meta = doc_meta          # doc_id -> info for display

    # --- the three methods Mapping requires; it derives the rest from these ----------
    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]        # KeyError for unknown terms, like a dict

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    # --- overrides ---------------------------------------------------------------------
    def __contains__(self, term: object) -> bool:
        # Mapping's version does `try: self[term] except KeyError`; a direct dict
        # lookup is faster and clearer.
        return term in self.postings

    def __eq__(self, other: object) -> bool:
        # Mapping.__eq__ would compare postings only, so two indexes with different
        # documents could compare equal. Compare everything instead.
        if not isinstance(other, Index):
            return NotImplemented
        return (self.postings, self.doc_lengths, self.doc_meta) == (
            other.postings, other.doc_lengths, other.doc_meta
        )

    __hash__ = None  # mutable contents + custom __eq__ => unhashable, like dict

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(terms={len(self):,}, docs={self.num_docs:,}, "
                f"avg_doc_length={self.avg_doc_length:.1f})")

    # --- statistics ranking needs ----------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        # O(num_docs) once, then a plain attribute lookup: BM25 reads it for every posting.
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    @property
    def has_positions(self) -> bool:
        """Was this index built with token positions? (Every posting has tf >= 1, so
        one posting is enough to tell.)"""
        for plist in self.postings.values():
            return bool(plist[0].positions)
        return False

    def df(self, term: str) -> int:
        """Document frequency: how many documents contain the term (0 if none)."""
        postings = self.postings.get(term)
        return len(postings) if postings is not None else 0


def build_index(docs: Iterable[Document], *, positions: bool = True) -> Index:
    """One streaming pass over the corpus.

    positions=True also records where in each document every term occurs, which
    phrase queries need. It costs memory: one int per token in the corpus.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):  # ascending ids => every list stays sorted for free
        doc_meta[doc_id] = DocMeta(ext_id=doc.doc_id, path=str(doc.path), title=doc.title)
        if positions:
            # term -> its offsets in this document; tf is just how many there are
            where: defaultdict[str, list[int]] = defaultdict(list)
            n_tokens = 0
            for n_tokens, term in enumerate(tokenize(doc.text), start=1):
                where[term].append(n_tokens - 1)
            doc_lengths[doc_id] = n_tokens
            for term, offs in where.items():
                postings[term].append(Posting(doc_id, len(offs), tuple(offs)))
        else:
            counts = Counter(tokenize(doc.text))  # this document's term frequencies, then discarded
            doc_lengths[doc_id] = sum(counts.values())
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
    ap.add_argument("--no-positions", action="store_true",
                    help="don't record token positions (smaller, but no phrase queries)")
    args = ap.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(args.root), args.limit),
                        positions=not args.no_positions)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_postings = sum(len(p) for p in index.values())
    print(f"documents:    {index.num_docs}")
    print(f"terms:        {len(index)}")
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
        plist = index.get(args.term, [])
        print(f"'{args.term}': in {len(plist)} documents; first 5:")
        for p in plist[:5]:
            print(f"  doc {p.doc_id} ({index.doc_meta[p.doc_id].title!r}) tf={p.tf} "
                  f"positions={p.positions[:10]}")


if __name__ == "__main__":
    # When run with `python -m findex.index`, this file is the module "__main__", so the
    # classes defined above would be pickled as "__main__.Posting" and could not be loaded
    # from anywhere else. Re-importing under the real name avoids that.
    from findex.index import main as _main

    _main()