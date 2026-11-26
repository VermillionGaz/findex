"""Ranked Boolean search over the in-memory index.

    python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'
    python -m findex.search index.bin "cat dog" --scorer tfidf              # TF-IDF instead of BM25
    python -m findex.search index.bin "cat dog" --scorer none               # unranked (Lab 2 style)
    python -m findex.search index.bin "cat dog" --explain                   # show the parsed tree
    python -m findex.search data/simplewiki.jsonl "cat dog" --limit 20000   # build first

Query language (see query.py): AND, OR, NOT (uppercase), parentheses, "quoted phrases";
a space means AND; precedence NOT > AND > OR.

The query decides WHICH documents match; the scorer decides their ORDER, using
every term that isn't negated.
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable
from functools import partial
from itertools import islice
from pathlib import Path

from .corpus import iter_documents
from .index import Index, build_index
from .query import ParseError, parse
from .rank import BM25, SCORERS, Scorer, SearchResult, rank
from .store import detect_format, load

DocIds = list[int]  # always sorted ascending, no duplicates


# --- two-pointer merges: O(len(a) + len(b)), no hashing ------------------------------

def and_merge(a: DocIds, b: DocIds) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def or_merge(a: DocIds, b: DocIds) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])  # one of these two is empty
    out.extend(b[j:])
    return out


def not_merge(a: DocIds, b: DocIds) -> DocIds:
    """a minus b."""
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


# --- set-based alternative: C-speed hashing, but must re-sort ------------------------

def and_set(a: DocIds, b: DocIds) -> DocIds:
    return sorted(set(a) & set(b))


def or_set(a: DocIds, b: DocIds) -> DocIds:
    return sorted(set(a) | set(b))


def not_set(a: DocIds, b: DocIds) -> DocIds:
    return sorted(set(a) - set(b))


Op = Callable[[DocIds, DocIds], DocIds]
ENGINES: dict[str, dict[str, Op]] = {
    "merge": {"AND": and_merge, "OR": or_merge, "NOT": not_merge},
    "set": {"AND": and_set, "OR": or_set, "NOT": not_set},
}


# --- query evaluation ------------------------------------------------------------------

def evaluate(index: Index, query: str, engine: str = "merge") -> DocIds:
    """Sorted ids of the documents matching the query.

    Lab 2 evaluated strictly left to right; now the query is parsed into a tree
    (with precedence and parentheses) and the tree evaluates itself, using the
    chosen engine's merge functions.
    """
    return parse(query).evaluate(index, ENGINES[engine])


def positive_terms(query: str) -> list[str]:
    """The tokens that contribute to the score: every term and phrase word except
    negated ones. Duplicates are kept, so "cat cat" weighs cat twice."""
    return parse(query).positive_terms()


def search(
    index: Index,
    query: str,
    scorer: Scorer = BM25(),
    k: int = 10,
    engine: str = "merge",
) -> list[SearchResult]:
    """Top-k documents matching the Boolean query, best first.

    (BM25() as a default argument is safe: default values are created once, but a
    frozen dataclass can't be mutated, so every call sees the same, unchanged scorer.)
    """
    tree = parse(query)
    candidates = set(tree.evaluate(index, ENGINES[engine]))
    terms = tree.positive_terms()
    if not terms:
        # e.g. "NOT cat": nothing to score by, so the matches come back unranked.
        return [SearchResult(0.0, d, index.doc_meta[d].title) for d in sorted(candidates)[:k]]
    return rank(index, terms, scorer, k, candidates)


# --- benchmark: merge vs set --------------------------------------------------------------

def best_of(fn: Callable[[], object], repeats: int = 7) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(index: Index) -> None:
    by_df = sorted(index, key=index.df)
    rare, common = by_df[:2], by_df[-2:]
    cases = {
        "2 most common": (common[0], common[1]),
        "2 rarest": (rare[0], rare[1]),
        "common x rare": (common[1], rare[0]),
    }
    print(f"{'case':<15}{'terms (doc freq)':<42}{'op':<5}{'merge µs':>11}{'set µs':>10}")
    for name, (t1, t2) in cases.items():
        a = [p.doc_id for p in index.get(t1, [])]
        b = [p.doc_id for p in index.get(t2, [])]
        label = f"{t1}({len(a)}), {t2}({len(b)})"
        for op in ("AND", "OR"):
            merge_fn, set_fn = ENGINES["merge"][op], ENGINES["set"][op]
            assert merge_fn(a, b) == set_fn(a, b)  # same answer
            m = best_of(partial(merge_fn, a, b)) * 1e6
            s = best_of(partial(set_fn, a, b)) * 1e6
            print(f"{name:<15}{label:<42}{op:<5}{m:>11.1f}{s:>10.1f}")

def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.search")
    ap.add_argument("source", type=Path, help="a saved index file, or a corpus to index")
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    ap.add_argument("--top", type=int, default=10, help="how many results to print")
    ap.add_argument("--scorer", choices=[*SCORERS, "none"], default="bm25",
                    help="ranking function; 'none' = unranked Boolean results (Lab 2)")
    ap.add_argument("--explain", action="store_true", help="print the parsed query tree")
    ap.add_argument("--bench", action="store_true", help="benchmark merge vs set")
    ap.add_argument("--trust-pickle", action="store_true", help="allow loading a pickle index (only files you created yourself!)")
    args = ap.parse_args()

    if not args.bench:
        try:
            tree = parse(args.query)
        except ParseError as e:            # check the query before loading anything
            ap.error(str(e))
        if args.explain:
            print(f"parsed: {tree}\n        {tree!r}")

    tracemalloc.start()
    t0 = time.perf_counter()
    if detect_format(args.source):
        index = load(args.source, trust_pickle=args.trust_pickle)
        print(f"loaded {index.num_docs} documents in {time.perf_counter() - t0:.2f}s")
    else:
        index = build_index(islice(iter_documents(args.source), args.limit))
        print(f"indexed {index.num_docs} documents in {time.perf_counter() - t0:.2f}s")

    if args.bench:
        tracemalloc.stop()  # tracing would distort the timings
        benchmark(index)
        return
    t0 = time.perf_counter()
    if args.scorer == "none":
        hits = evaluate(index, args.query, args.engine)
        results = [SearchResult(0.0, d, index.doc_meta[d].title) for d in hits[: args.top]]
        summary = f"{len(hits)} results"
    else:
        results = search(index, args.query, SCORERS[args.scorer], args.top, args.engine)
        summary = f"top {len(results)} results"
    ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{summary} for {args.query!r} ({args.engine}, {args.scorer}, {ms:.2f} ms)")
    print(f"peak memory: {peak / 1e6:.1f} MB")
    for r in results:
        print(f"  {r.score:7.3f}  [{r.doc_id}] {r.title}  ({index.doc_meta[r.doc_id].ext_id})")


if __name__ == "__main__":
    main()