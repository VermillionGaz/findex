"""Chunked indexing (build_partial + merge) must produce exactly the index of one
serial pass (Lab 4's build_index), however the corpus is cut into chunks."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest

from findex.corpus import TxtChunk, iter_documents, plan_chunks, read_chunk
from findex.index import Index, build_index
from findex.parallel import build_index_parallel, build_partial, build_partials, merge
from findex.store import save


def saved_bytes(index: Index) -> bytes:
    """The index as saved to disk: byte equality also checks term and document order,
    which == on two Index objects does not (dict equality ignores order)."""
    with tempfile.TemporaryDirectory() as tmp:
        save(index, Path(tmp) / "i.bin")
        return (Path(tmp) / "i.bin").read_bytes()


def chunked(corpus: Path, n_chunks: int, positions: bool = True, limit: int | None = None) -> Index:
    return merge(build_partial(c, positions) for c in plan_chunks(corpus, n_chunks, limit))


@pytest.mark.parametrize("positions", [True, False])
def test_serial_build_is_identical_to_lab4(corpus_jsonl: Path, positions: bool) -> None:
    """The lab's required test: merge([build_partial(everything)]) == build_index(...)."""
    lab4 = build_index(iter_documents(corpus_jsonl), positions=positions)
    serial = merge([build_partial(plan_chunks(corpus_jsonl, 1)[0], positions)])
    assert serial == lab4
    assert saved_bytes(serial) == saved_bytes(lab4)


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 5, 8, 50])
def test_any_number_of_chunks_gives_the_same_index(
    corpus_jsonl: Path, corpus_dir: Path, n_chunks: int
) -> None:
    for corpus in (corpus_jsonl, corpus_dir):
        lab4 = build_index(iter_documents(corpus))
        assert saved_bytes(chunked(corpus, n_chunks)) == saved_bytes(lab4)


def test_limit(corpus_jsonl: Path) -> None:
    for limit in (0, 1, 5, 100):
        docs = list(iter_documents(corpus_jsonl))[:limit]
        assert chunked(corpus_jsonl, 3, limit=limit) == build_index(docs)


def test_chunks_cover_the_corpus_exactly_once(corpus_jsonl: Path) -> None:
    chunks = plan_chunks(corpus_jsonl, 4)
    assert [c.start for c in chunks[1:]] == [c.end for c in chunks[:-1]]  # contiguous
    every = [d for c in chunks for d in read_chunk(c)]
    assert every == list(iter_documents(corpus_jsonl))


def test_partial_index_is_compact_and_picklable(corpus_jsonl: Path) -> None:
    part = build_partial(plan_chunks(corpus_jsonl, 1)[0])
    assert pickle.loads(pickle.dumps(part)) == part
    assert part.num_docs == 8 and sum(part.df) == len(part.doc_ids) == len(part.tfs)
    assert part.positions is not None and len(part.positions) == sum(part.tfs)


def test_empty_corpus(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert chunked(empty, 4) == build_index([])


# --- M2: the executors ---------------------------------------------------------------------


@pytest.mark.slow  # starting worker processes ("spawn") takes about a second
@pytest.mark.parametrize(
    ("executor", "workers"), [("threads", 1), ("threads", 3), ("processes", 2)]
)
def test_every_executor_gives_the_same_index(
    corpus_jsonl: Path, corpus_dir: Path, executor, workers: int
) -> None:
    for corpus in (corpus_jsonl, corpus_dir):
        lab4 = build_index(iter_documents(corpus))
        parallel = build_index_parallel(corpus, executor, workers)
        assert saved_bytes(parallel) == saved_bytes(lab4)


def test_results_come_back_in_chunk_order(corpus_jsonl: Path) -> None:
    chunks = plan_chunks(corpus_jsonl, 8)
    partials = build_partials(chunks, "threads", workers=4)
    assert [p.doc_meta for p in partials] == [build_partial(c).doc_meta for c in chunks]


def failing_chunks(root: Path, good: list) -> list:
    """A missing file makes build_partial raise FileNotFoundError inside the worker."""
    return [TxtChunk(root, (root / "does-not-exist.txt",)), *good]


@pytest.mark.parametrize(
    "executor", ["serial", "threads", pytest.param("processes", marks=pytest.mark.slow)]
)
def test_a_failing_worker_raises_its_original_exception(corpus_dir: Path, executor) -> None:
    chunks = failing_chunks(corpus_dir, plan_chunks(corpus_dir, 4))
    with pytest.raises(FileNotFoundError, match="does-not-exist") as info:
        build_partials(chunks, executor, workers=2)
    if executor == "processes":
        # The worker's own traceback travels back with the exception, chained as its cause.
        assert "build_partial" in str(info.value.__cause__)


def test_a_failure_cancels_the_chunks_still_waiting(corpus_dir: Path) -> None:
    good = plan_chunks(corpus_dir, 8) * 5  # 40 more chunks queued behind the failing one
    finished: list[object] = []
    with pytest.raises(FileNotFoundError):
        build_partials(
            failing_chunks(corpus_dir, good), "threads", 1, on_chunk_done=finished.append
        )
    assert len(finished) <= 1  # at most the chunk already running; the rest never started


def test_corpus_bytes(corpus_jsonl: Path, corpus_dir: Path) -> None:
    from findex.corpus import corpus_bytes

    assert corpus_bytes(corpus_jsonl) == corpus_jsonl.stat().st_size
    first_line = corpus_jsonl.read_bytes().split(b"\n")[0]
    assert corpus_bytes(corpus_jsonl, limit=1) == len(first_line) + 1  # + its newline
    assert corpus_bytes(corpus_dir) == sum(p.stat().st_size for p in corpus_dir.rglob("*.txt"))
    assert corpus_bytes(corpus_dir, limit=0) == 0
