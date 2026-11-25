"""Boolean search over the in-memory index.

    python -m findex.search data/simplewiki.jsonl "cat dog" --limit 20000
    python -m findex.search data/simplewiki.jsonl "cat OR dog NOT bird" --engine set
    python -m findex.search data/simplewiki.jsonl --bench --limit 20000

Query language (operators must be UPPERCASE; evaluated strictly left to right):
    a b        implicit AND
    a AND b    explicit AND
    a OR b     union
    a NOT b    a AND NOT b  (documents with a but without b)
    NOT a      every document without a
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from itertools import islice
from pathlib import Path

from .corpus import iter_documents
from .index import Index, build_index
from .tokenize import tokenize

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

def term_ids(index: Index, word: str, ops: dict[str, Op]) -> DocIds | None:
    """Doc ids for one query word, normalized exactly like the corpus was.

    "Cats" -> "cats"; "U.S." -> two tokens u, s -> must contain both.
    Returns None if the word has no tokens at all (e.g. "!!!"): it is ignored.
    """
    result: DocIds | None = None
    for tok in tokenize(word):
        ids = [p.doc_id for p in index.get(tok)]
        result = ids if result is None else ops["AND"](result, ids)
    return result


def evaluate(index: Index, query: str, engine: str = "merge") -> DocIds:
    ops = ENGINES[engine]
    result: DocIds | None = None
    pending = "AND"          # operator waiting for its right-hand side
    expect_term = False      # True right after an operator

    for word in query.split():
        if word in ("AND", "OR", "NOT"):
            if expect_term:
                raise ValueError(f"operator {word!r} directly after another operator")
            if word == "NOT":
                pending = "NOT" if pending in ("AND", "NOT") else _bad_not()
            elif result is None:
                raise ValueError(f"query cannot start with {word!r}")
            else:
                pending = word
            expect_term = True
            continue

        ids = term_ids(index, word, ops)
        if ids is None:
            continue
        if result is None:
            # first operand: "NOT a" means "all documents minus a"
            result = ops["NOT"](list(range(index.n_docs)), ids) if pending == "NOT" else ids
        else:
            result = ops[pending](result, ids)
        pending = "AND"
        expect_term = False

    if expect_term:
        raise ValueError("query ends with an operator")
    return result if result is not None else []


def _bad_not() -> str:
    raise ValueError("'OR NOT' is not supported; write 'a NOT b' instead")


# --- benchmark: merge vs set --------------------------------------------------------------

def best_of(fn: Callable[[], object], repeats: int = 7) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark(index: Index) -> None:
    by_df = sorted(index.postings, key=lambda t: len(index.postings[t]))
    rare, common = by_df[:2], by_df[-2:]
    cases = {
        "2 most common": (common[0], common[1]),
        "2 rarest": (rare[0], rare[1]),
        "common x rare": (common[1], rare[0]),
    }
    print(f"{'case':<15}{'terms (doc freq)':<42}{'op':<5}{'merge µs':>11}{'set µs':>10}")
    for name, (t1, t2) in cases.items():
        a = [p.doc_id for p in index.get(t1)]
        b = [p.doc_id for p in index.get(t2)]
        label = f"{t1}({len(a)}), {t2}({len(b)})"
        for op in ("AND", "OR"):
            merge_fn, set_fn = ENGINES["merge"][op], ENGINES["set"][op]
            assert merge_fn(a, b) == set_fn(a, b)  # same answer
            m = best_of(partial(merge_fn, a, b)) * 1e6
            s = best_of(partial(set_fn, a, b)) * 1e6
            print(f"{name:<15}{label:<42}{op:<5}{m:>11.1f}{s:>10.1f}")

def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.search")
    ap.add_argument("root", type=Path, help="corpus to index (a saved index comes in M3)")
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    ap.add_argument("--top", type=int, default=10, help="how many results to print")
    ap.add_argument("--bench", action="store_true", help="benchmark merge vs set")
    args = ap.parse_args()

    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(args.root), args.limit))
    print(f"indexed {index.n_docs} documents in {time.perf_counter() - t0:.2f}s")

    if args.bench:
        benchmark(index)
        return
    t0 = time.perf_counter()
    hits = evaluate(index, args.query, args.engine)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{len(hits)} results for {args.query!r} ({args.engine}, {ms:.2f} ms)")
    for doc_id in hits[: args.top]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id}] {meta.title}  ({meta.ext_id})")


if __name__ == "__main__":
    main()