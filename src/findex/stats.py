"""python -m findex.stats data/simplewiki.jsonl [--limit N] [--eager]"""
from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from .corpus import iter_documents
from .tokenize import tokenize


def stats_lazy(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    docs = islice(iter_documents(root), limit)  # islice(x, None) == no limit
    n_docs = 0
    n_tokens = 0
    counts: Counter[str] = Counter()  # the ONLY thing that grows: it is the answer

    def tokens() -> Iterable[str]:
        nonlocal n_docs
        for doc in docs:
            n_docs += 1
            yield from tokenize(doc.text)

    for tok in tokens():
        n_tokens += 1
        counts[tok] += 1
    return n_docs, n_tokens, counts


def stats_eager(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    """Deliberately wasteful baseline for the lab's comparison table."""
    docs = list(islice(iter_documents(root), limit))            # all texts in RAM
    all_tokens = [t for d in docs for t in tokenize(d.text)]    # all tokens in RAM
    return len(docs), len(all_tokens), Counter(all_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.stats")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eager", action="store_true", help="use the memory-hungry version")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    fn = stats_eager if args.eager else stats_lazy
    tracemalloc.start()
    t0 = time.perf_counter()
    n_docs, n_tokens, counts = fn(args.root, args.limit)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mode:        {'eager' if args.eager else 'lazy'}")
    print(f"documents:   {n_docs}")
    print(f"tokens:      {n_tokens}")
    print(f"vocabulary:  {len(counts)}")
    print(f"elapsed:     {elapsed:.2f}s")
    print(f"peak memory: {peak / 1e6:.1f} MB")
    print(f"top {args.top}:")
    for term, c in counts.most_common(args.top):
        print(f"  {c:>8}  {term}")


if __name__ == "__main__":
    main()