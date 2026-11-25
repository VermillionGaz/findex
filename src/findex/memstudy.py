"""M4 -- the memory study.

Builds the postings three ways over the SAME streamed corpus and measures each:

  1. plain   : list[PlainPosting]  -- @dataclass(frozen=True), no slots
  2. slots   : list[SlotPosting]   -- @dataclass(frozen=True, slots=True), like index.Posting
  3. arrays  : (array('I') doc_ids, array('I') tfs) per term -- no objects at all

For each: peak memory during build, memory retained after build, bytes per
posting, pickle file size and pickle load time. All three use pickle, so the
file/load columns compare *representations*, not file formats.

    python -m findex.memstudy data/simplewiki.jsonl [--limit N] [--out-dir DIR]

Prints a Markdown table to paste into the README.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .corpus import Document, iter_documents
from .tokenize import tokenize

assert array("I").itemsize == 4, "this study needs 4-byte unsigned ints"


# --- the two object-based Posting variants ----------------------------------------------
# Identical except for slots=True, so any difference between them is the slots alone.
@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


# --- builders: one streaming pass each, same id scheme as index.build_index -------------
def _term_counts(docs: Iterable[Document]) -> Iterator[tuple[int, Counter[str]]]:
    # Ascending internal ids (0, 1, 2, ...) => every postings list stays sorted for free.
    for doc_id, doc in enumerate(docs):
        yield doc_id, Counter(tokenize(doc.text))


def build_objects(docs: Iterable[Document], cls: type) -> dict[str, list]:
    postings: dict[str, list] = {}
    for doc_id, counts in _term_counts(docs):
        for term, tf in counts.items():
            postings.setdefault(term, []).append(cls(doc_id, tf))
    return postings


def build_arrays(docs: Iterable[Document]) -> dict[str, tuple[array, array]]:
    postings: dict[str, tuple[array, array]] = {}
    for doc_id, counts in _term_counts(docs):
        for term, tf in counts.items():
            pair = postings.get(term)
            if pair is None:
                pair = postings[term] = (array("I"), array("I"))
            pair[0].append(doc_id)
            pair[1].append(tf)
    return postings


VARIANTS: dict[str, tuple[str, Callable[[Iterable[Document]], dict]]] = {
    "plain": ("`list[Posting]` with plain `@dataclass`", lambda d: build_objects(d, PlainPosting)),
    "slots": ("`list[Posting]` with `slots=True`", lambda d: build_objects(d, SlotPosting)),
    "arrays": ("`array('I')` pairs (doc_ids, tfs) — no objects", build_arrays),
}


def as_pairs(postings: dict) -> dict[str, list[tuple[int, int]]]:
    """Normalize any variant to term -> [(doc_id, tf)] so variants can be compared."""
    out = {}
    for term, v in postings.items():
        out[term] = list(zip(*v)) if isinstance(v, tuple) else [(p.doc_id, p.tf) for p in v]
    return out


def count_postings(postings: dict) -> int:
    return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in postings.values())


# --- measurement --------------------------------------------------------------------------
def measure(name: str, docs: Iterable[Document], out_dir: Path) -> dict:
    _, build = VARIANTS[name]
    gc.collect()

    tracemalloc.start()
    postings = build(docs)
    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = out_dir / f"memstudy_{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(postings, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = path.stat().st_size

    gc.collect()
    t0 = time.perf_counter()
    with path.open("rb") as f:
        pickle.load(f)  # our own file, written just above -- the only safe kind of pickle
    load_s = time.perf_counter() - t0
    path.unlink()

    return {
        "name": name,
        "postings": postings,
        "n_postings": count_postings(postings),
        "n_terms": len(postings),
        "peak": peak,
        "retained": retained,
        "size": size,
        "load_s": load_s,
    }


def _real_bytes_each(make: Callable[[int], object], n: int = 100_000) -> float:
    """Actual allocated bytes per element (incl. its list slot), via tracemalloc.

    Unlike sys.getsizeof, this sees allocator rounding and the doc_id int that is
    created alongside each posting.
    """
    gc.collect()
    tracemalloc.start()
    xs = [make(10**6 + i) for i in range(n)]
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del xs
    return current / n


def per_object_sizes() -> str:
    """Per-element costs that explain the bytes-per-posting column.

    Don't use sys.getsizeof(obj.__dict__) for this: since CPython 3.11 a plain
    instance stores its attribute values inline and only builds a real dict if
    something asks for __dict__ -- so asking for it creates the very cost you
    were trying to measure.
    """
    plain = _real_bytes_each(lambda i: PlainPosting(i, 3))
    slots = _real_bytes_each(lambda i: SlotPosting(i, 3))
    return (
        f"- plain `@dataclass` posting: {plain:.0f} B each, measured "
        f"(8 B list pointer + {sys.getsizeof(10**6)} B doc_id int + instance with inline values)\n"
        f"- `slots=True` posting: {slots:.0f} B each, measured "
        f"(same, but the instance is just a header + 2 fixed slots)\n"
        f"- `array('I')` posting: 8 B (two raw 4-byte ints: no objects, no pointers)\n"
        f"- ints -5..256 (most tf values, the first 257 doc ids) are cached singletons: ~0 B"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.memstudy")
    ap.add_argument("root", type=Path, help="corpus: a .jsonl file or a folder of .txt files")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the temporary pickles (default: a temp dir)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.out_dir or Path(tmp)
        results = []
        for name in VARIANTS:
            # A fresh stream per variant: generators can only be consumed once.
            results.append(measure(name, islice(iter_documents(args.root), args.limit), out_dir))
            print(f"  {name}: done", file=sys.stderr)

    # Sanity check: all three representations must hold exactly the same postings.
    reference = as_pairs(results[0]["postings"])
    for r in results[1:]:
        assert as_pairs(r["postings"]) == reference, f"{r['name']} differs from plain"
    for r in results:
        del r["postings"]

    n = results[0]["n_postings"]
    print(f"\nCorpus: {args.root} (limit={args.limit}), "
          f"{results[0]['n_terms']:,} terms, {n:,} postings, CPython {sys.version.split()[0]}\n")
    print("| Postings representation | Peak memory (build) | Retained after build "
          "| Bytes / posting | Index file size | Load time |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {VARIANTS[r['name']][0]} | {r['peak'] / 1e6:.1f} MB "
              f"| {r['retained'] / 1e6:.1f} MB | {r['retained'] / n:.1f} "
              f"| {r['size'] / 1e6:.1f} MB | {r['load_s']:.3f} s |")
    print("\nPer-posting costs:")
    print(per_object_sizes())


if __name__ == "__main__":
    # Same trick as index.py: under `python -m`, this file is "__main__", so the
    # classes above would pickle as "__main__.PlainPosting". Re-import under the
    # real name so the pickles refer to findex.memstudy.PlainPosting.
    from findex.memstudy import main as _main

    _main()
