"""Indexing in parts: build a partial index per chunk of the corpus, then merge them.

    partials = [build_partial(chunk) for chunk in plan_chunks(corpus, n)]
    index = merge(partials)

Each partial numbers its documents 0, 1, 2, ... locally. merge() shifts them by the
number of documents in all earlier chunks, so the global ids are 0..N-1 in corpus
order -- exactly the ids a single pass over the corpus would assign. Because chunks
are contiguous and processed in order, every term's postings from chunk k come after
those from chunk k-1: the "sorted merge" of postings lists is plain concatenation.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass

from .corpus import Chunk, read_chunk
from .index import DocMeta, Index, Posting
from .timing import timed
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class PartialIndex:
    """The index of one chunk, as flat arrays rather than millions of Posting objects.

    With a process pool every partial is pickled in the worker and unpickled in the
    parent. Arrays pickle as a few blocks of raw bytes; a list of Posting objects
    would pickle one object at a time (see Lab 2's memory study).

    Same layout as the binary index file: postings are stored term by term, in the
    order the terms first appear; posting i owns the next tfs[i] numbers of positions.
    """

    terms: list[str]  # in order of first appearance in the chunk
    df: array[int]  # number of postings of each term
    doc_ids: array[int]  # local doc ids (0 = the chunk's first document)
    tfs: array[int]
    positions: array[int] | None  # None: built without positions
    doc_lengths: array[int]  # tokens per document, by local id
    doc_meta: list[tuple[str, str, str]]  # (ext_id, path, title) by local id

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)


@timed
def build_partial(chunk: Chunk, positions: bool = True) -> PartialIndex:
    """Index one chunk. A module-level function of picklable arguments, so a process
    pool can run it: the worker receives the small Chunk, not the documents."""
    # term -> (doc_ids, tfs, positions) of its postings, while building
    acc: dict[str, tuple[array[int], array[int], array[int]]] = {}
    doc_lengths = array("I")
    doc_meta: list[tuple[str, str, str]] = []

    for doc_id, doc in enumerate(read_chunk(chunk)):
        doc_meta.append((doc.doc_id, str(doc.path), doc.title))
        where: dict[str, list[int]] = {}  # term -> its offsets in this document
        n_tokens = 0
        for n_tokens, term in enumerate(tokenize(doc.text), start=1):
            where.setdefault(term, []).append(n_tokens - 1)
        doc_lengths.append(n_tokens)
        for term, offs in where.items():
            entry = acc.get(term)
            if entry is None:
                entry = acc[term] = (array("I"), array("I"), array("I"))
            entry[0].append(doc_id)
            entry[1].append(len(offs))
            if positions:
                entry[2].extend(offs)

    # Flatten into the compact layout, keeping first-appearance order of terms.
    terms = list(acc)
    df = array("I", (len(acc[t][0]) for t in terms))
    doc_ids, tfs, pos = array("I"), array("I"), array("I")
    for t in terms:
        d, f, p = acc[t]
        doc_ids.extend(d)
        tfs.extend(f)
        pos.extend(p)
    return PartialIndex(terms, df, doc_ids, tfs, pos if positions else None, doc_lengths, doc_meta)


@timed
def merge(partials: Iterable[PartialIndex]) -> Index:
    """Combine partial indexes, given in corpus order, into one Index.

    This runs in one process, after the workers finish: whatever it costs is the
    serial part of parallel indexing (Amdahl's law).
    """
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    offset = 0  # documents in all earlier chunks

    for part in partials:
        i = 0  # index into doc_ids / tfs
        off = 0  # index into positions
        pos = part.positions
        for term, n in zip(part.terms, part.df, strict=True):
            plist = postings.get(term)
            if plist is None:
                plist = postings[term] = []  # first seen here: same order as one pass
            for d, tf in zip(part.doc_ids[i : i + n], part.tfs[i : i + n], strict=True):
                if pos is None:
                    plist.append(Posting(d + offset, tf))
                else:
                    plist.append(Posting(d + offset, tf, tuple(pos[off : off + tf])))
                    off += tf
            i += n
        for local, (length, meta) in enumerate(zip(part.doc_lengths, part.doc_meta, strict=True)):
            doc_lengths[offset + local] = length
            doc_meta[offset + local] = DocMeta(*meta)
        offset += part.num_docs

    return Index(postings, doc_lengths, doc_meta)
