"""Save and load an Index in two formats.

pickle  - one call each way, but SEE THE WARNING BELOW.
binary  - a small JSON header plus raw 4-byte integer arrays (array('I')).

    python -m findex.store data/simplewiki.jsonl --limit 20000     # benchmark both
"""
from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
import tempfile
import time
from array import array
from functools import partial
from itertools import islice
from pathlib import Path
from typing import BinaryIO

from .corpus import iter_documents
from .index import DocMeta, Index, Posting, build_index

# ---------------------------------------------------------------------------------------
# WARNING: pickle.load() can execute arbitrary code. A malicious pickle file runs its
# code the moment you load it, before you ever look at the result. Only load pickle
# files that YOU created on this machine. Never load one that was downloaded, emailed
# or shared. That is why load() refuses pickle files unless trust_pickle=True.
# ---------------------------------------------------------------------------------------

FORMATS = ("pickle", "binary")
MAGIC = b"FINDEXB1"            # 8 bytes: identifies our binary format, version 1
assert array("I").itemsize == 4, "this format needs 4-byte unsigned ints"


def detect_format(path: str | Path) -> str | None:
    """Look at the first bytes of a file. Returns 'binary', 'pickle' or None."""
    with Path(path).open("rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        return "binary"
    if head[:1] == b"\x80":     # every pickle protocol >= 2 starts with this opcode
        return "pickle"
    return None


# --- pickle -----------------------------------------------------------------------------

def _save_pickle(index: Index, f: BinaryIO) -> None:
    pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(f: BinaryIO) -> Index:
    obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise ValueError("pickle file does not contain an Index")
    return obj


# --- binary ---------------------------------------------------------------------------------

def _write_array(arr: array, f: BinaryIO) -> None:
    if sys.byteorder == "big":   # the file is always little-endian
        arr = array("I", arr)
        arr.byteswap()
    arr.tofile(f)


def _read_array(f: BinaryIO, n: int) -> array:
    arr = array("I")
    try:
        arr.fromfile(f, n)
    except EOFError as exc:
        raise ValueError("index file is truncated") from exc
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def _save_binary(index: Index, f: BinaryIO) -> None:
    n_docs = index.n_docs
    if set(index.doc_meta) != set(range(n_docs)) or set(index.doc_lengths) != set(range(n_docs)):
        raise ValueError("doc ids must be exactly 0..n-1")

    terms = list(index.postings)
    df = array("I", (len(index.postings[t]) for t in terms))   # postings per term
    doc_ids = array("I")                                        # all postings, term by term
    tfs = array("I")
    for term in terms:
        for p in index.postings[term]:
            doc_ids.append(p.doc_id)
            tfs.append(p.tf)
    lengths = array("I", (index.doc_lengths[i] for i in range(n_docs)))

    header = {
        "version": 1,
        "terms": terms,
        "meta": [[m.ext_id, m.path, m.title] for m in (index.doc_meta[i] for i in range(n_docs))],
        "n_postings": len(doc_ids),
    }
    blob = json.dumps(header, ensure_ascii=False).encode("utf-8")
    f.write(MAGIC)
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    for arr in (lengths, df, doc_ids, tfs):
        _write_array(arr, f)


def _load_binary(f: BinaryIO) -> Index:
    if f.read(len(MAGIC)) != MAGIC:
        raise ValueError("not a findex binary file")
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen).decode("utf-8"))
    if header.get("version") != 1:
        raise ValueError(f"unsupported index version: {header.get('version')}")

    terms: list[str] = header["terms"]
    meta = header["meta"]
    lengths = _read_array(f, len(meta))
    df = _read_array(f, len(terms))
    doc_ids = _read_array(f, header["n_postings"])
    tfs = _read_array(f, header["n_postings"])
    if sum(df) != header["n_postings"]:
        raise ValueError("corrupt index: posting counts do not add up")

    postings: dict[str, list[Posting]] = {}
    pos = 0
    for term, n in zip(terms, df, strict=True):
        chunk = zip(doc_ids[pos:pos + n], tfs[pos:pos + n], strict=True)
        postings[term] = [Posting(d, t) for d, t in chunk]
        pos += n
    return Index(
        postings=postings,
        doc_lengths=dict(enumerate(lengths)),
        doc_meta={i: DocMeta(*m) for i, m in enumerate(meta)},
    )


# --- public API ---------------------------------------------------------------------------

def save(index: Index, path: str | Path, fmt: str = "binary") -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    with Path(path).open("wb") as f:
        (_save_pickle if fmt == "pickle" else _save_binary)(index, f)


def load(path: str | Path, *, trust_pickle: bool = False) -> Index:
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"{path} is not a findex index file")
    if fmt == "pickle" and not trust_pickle:
        raise PermissionError(
            "this is a pickle file, and loading a pickle can run arbitrary code. "
            "Pass trust_pickle=True (CLI: --trust-pickle) only if you created it yourself."
        )
    with Path(path).open("rb") as f:
        return _load_pickle(f) if fmt == "pickle" else _load_binary(f)


# --- benchmark -------------------------------------------------------------------------------

def _best(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(prog="findex.store")
    ap.add_argument("root", type=Path, help="corpus to index")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    index = build_index(islice(iter_documents(args.root), args.limit))
    print(f"indexed {index.n_docs} documents, {len(index.postings)} terms\n")
    print(f"{'format':<10}{'size MB':>10}{'save s':>10}{'load s':>10}  round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in FORMATS:
            path = Path(tmp) / f"index.{fmt}"
            t_save = _best(partial(save, index, path, fmt))
            t_load = _best(partial(load, path, trust_pickle=True))
            verdict = "OK" if load(path, trust_pickle=True) == index else "MISMATCH"
            size = path.stat().st_size / 1e6
            print(f"{fmt:<10}{size:>10.1f}{t_save:>10.2f}{t_load:>10.2f}  {verdict}")


if __name__ == "__main__":
    main()