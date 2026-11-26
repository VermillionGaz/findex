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
import logging
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from functools import lru_cache, partial
from itertools import islice
from pathlib import Path
from typing import Literal

from .corpus import iter_documents
from .index import Index, build_index
from .query import OpName, ParseError, Query, parse
from .rank import DEFAULT_SCORER, SCORERS, Scorer, SearchResult, rank
from .snippets import ANSI, PLAIN, fetch_texts, make_snippet
from .store import detect_format, load
from .timing import setup_logging, timed

# Not __name__: under `python -m findex.search` that would be "__main__".
log = logging.getLogger("findex.search")

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


type Engine = Literal["merge", "set"]
Op = Callable[[DocIds, DocIds], DocIds]
ENGINES: dict[Engine, dict[OpName, Op]] = {
    "merge": {"AND": and_merge, "OR": or_merge, "NOT": not_merge},
    "set": {"AND": and_set, "OR": or_set, "NOT": not_set},
}


# --- query evaluation ------------------------------------------------------------------


def evaluate(index: Index, query: str, engine: Engine = "merge") -> DocIds:
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


class Searcher:
    """Searches one index and remembers recent query results.

    The cache maps a parsed query tree to its matching doc ids. Why the tree and
    not the query string? Trees are frozen dataclasses, so they're hashable and
    equal by value: "cat dog", "Cat  dog" and "cat AND dog" all parse to the
    same tree and share one cache entry.

    Why not @lru_cache on a plain function? lru_cache keys on ALL arguments, so
    the index would have to be one -- but Index is unhashable (and shouldn't be a
    cache key anyway). Instead every Searcher wraps its own bound method in its
    own cache, so each index gets a separate one, freed with the Searcher.
    """

    def __init__(self, index: Index, engine: Engine = "merge", cache_size: int = 256) -> None:
        self.index = index
        self.engine: Engine = engine
        self._match = lru_cache(maxsize=cache_size)(self._match_uncached)
        self.last_total = 0  # how many documents matched the most recent search

    def _match_uncached(self, tree: Query) -> tuple[int, ...]:
        # A tuple, not a list: the cached object is shared by every later caller,
        # so it must be immutable or one caller could corrupt it for the rest.
        return tuple(tree.evaluate(self.index, ENGINES[self.engine]))

    def matching(self, tree: Query) -> tuple[int, ...]:
        """Sorted ids of the documents matching a parsed query (cached)."""
        hits_before = self._match.cache_info().hits
        ids = self._match(tree)
        info = self._match.cache_info()
        log.debug(
            "query %s: cache %s (%d hits, %d misses, %d/%d entries)",
            tree,
            "HIT" if info.hits > hits_before else "miss",
            info.hits,
            info.misses,
            info.currsize,
            info.maxsize,
        )
        return ids

    @timed
    def search(
        self, query: str, scorer: Scorer | None = DEFAULT_SCORER, k: int = 10
    ) -> list[SearchResult]:
        """Top-k documents matching the query, best first.

        scorer=None returns the first k matches unranked (score 0), like Lab 2.
        (A default value is created once, when `def` runs, and shared by every call.
        That's only safe for immutable objects -- hence the frozen DEFAULT_SCORER,
        not a fresh-looking `BM25()` in the signature.)
        """
        tree = parse(query)
        ids = self.matching(tree)
        self.last_total = len(ids)
        terms = tree.positive_terms()
        if scorer is None or not terms:
            # Unranked: either asked for, or nothing to score by (e.g. "NOT cat").
            return [SearchResult(0.0, d, self.index.doc_meta[d].title) for d in ids[:k]]
        return rank(self.index, terms, scorer, k, set(ids))

    def cache_clear(self) -> None:
        self._match.cache_clear()


def search(
    index: Index,
    query: str,
    scorer: Scorer | None = DEFAULT_SCORER,
    k: int = 10,
    engine: Engine = "merge",
) -> list[SearchResult]:
    """One-off search without keeping a cache around (tests, scripts)."""
    return Searcher(index, engine, cache_size=0).search(query, scorer, k)


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


def _queries(args: argparse.Namespace) -> Iterator[str]:
    yield from args.query
    if args.interactive:
        while True:
            try:
                q = input("query> ").strip()
            except EOFError:  # Ctrl-D (Ctrl-Z Enter on Windows)
                return
            if not q:
                return
            yield q


def _show(searcher: Searcher, query: str, args: argparse.Namespace) -> None:
    try:
        tree = parse(query)
    except ParseError as e:
        log.error("%s", e)
        return
    if args.explain:
        print(f"parsed: {tree}\n        {tree!r}")
    t0 = time.perf_counter()
    try:
        results = searcher.search(query, SCORERS.get(args.scorer), args.top)
    except ValueError as e:  # e.g. a phrase on an index without positions
        log.error("%s", e)
        return
    ms = (time.perf_counter() - t0) * 1000
    print(
        f"\n{searcher.last_total} matches for {query!r}, showing {len(results)} "
        f"({args.scorer}, {args.engine}, {ms:.2f} ms)"
    )

    texts = fetch_texts(searcher.index, [r.doc_id for r in results]) if args.snippets else {}
    mark = ANSI if sys.stdout.isatty() else PLAIN
    terms = tree.positive_terms()
    for r in results:
        print(
            f"  {r.score:7.3f}  [{r.doc_id}] {r.title}  ({searcher.index.doc_meta[r.doc_id].ext_id})"
        )
        if r.doc_id in texts:
            print(f"           {make_snippet(texts[r.doc_id], terms, mark=mark)}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.search")
    ap.add_argument("source", type=Path, help="a saved index file, or a corpus to index")
    ap.add_argument("query", nargs="*", help="one or more queries (quote each one)")
    ap.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="after the given queries, keep reading queries from the keyboard",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    ap.add_argument("--top", type=int, default=10, help="how many results to print")
    ap.add_argument(
        "--scorer",
        choices=[*SCORERS, "none"],
        default="bm25",
        help="ranking function; 'none' = unranked Boolean results (Lab 2)",
    )
    ap.add_argument(
        "--no-snippets",
        dest="snippets",
        action="store_false",
        help="don't re-read the corpus to show highlighted snippets",
    )
    ap.add_argument("--explain", action="store_true", help="print the parsed query tree")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the timing and cache log")
    ap.add_argument(
        "--memory",
        action="store_true",
        help="measure peak memory while loading (tracemalloc makes loading ~10x slower)",
    )
    ap.add_argument("--bench", action="store_true", help="benchmark merge vs set")
    ap.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow loading a pickle index (only files you created yourself!)",
    )
    args = ap.parse_args()
    setup_logging(args.verbose)
    if not (args.query or args.interactive or args.bench):
        ap.error("give a query, -i for interactive mode, or --bench")
    for q in args.query:  # check the queries before loading anything
        try:
            parse(q)
        except ParseError as e:
            ap.error(str(e))

    if args.memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    if detect_format(args.source):
        index = load(args.source, trust_pickle=args.trust_pickle)
        log.info("loaded %d documents in %.2fs", index.num_docs, time.perf_counter() - t0)
    else:
        index = build_index(islice(iter_documents(args.source), args.limit))
        log.info("indexed %d documents in %.2fs", index.num_docs, time.perf_counter() - t0)
    if args.memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # tracing slows everything down; stop it now
        print(f"peak memory while loading: {peak / 1e6:.1f} MB")

    if args.bench:
        benchmark(index)
        return
    searcher = Searcher(index, args.engine)
    for query in _queries(args):
        _show(searcher, query, args)


if __name__ == "__main__":
    main()
