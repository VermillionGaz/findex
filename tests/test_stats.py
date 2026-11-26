"""Lab 1's corpus statistics: the lazy and eager versions, and their agreement with the
numbers `findex stats` computes from an index."""

from pathlib import Path

import pytest

from findex.index import Index
from findex.stats import stats_eager, stats_lazy


def test_lazy_and_eager_agree(corpus_jsonl: Path) -> None:
    assert stats_lazy(corpus_jsonl) == stats_eager(corpus_jsonl)


@pytest.mark.parametrize("limit", [0, 1, 3, None])
def test_limit(corpus_jsonl: Path, corpus: dict[str, str], limit: int | None) -> None:
    n_docs, _, _ = stats_lazy(corpus_jsonl, limit)
    assert n_docs == (len(corpus) if limit is None else limit)


def test_index_statistics_match_the_corpus_stream(corpus_jsonl: Path, tiny_index: Index) -> None:
    """`findex stats` reads its numbers from the index; Lab 1 streamed the corpus.
    Two independent routes to the same numbers must agree."""
    n_docs, n_tokens, counts = stats_lazy(corpus_jsonl)
    assert tiny_index.num_docs == n_docs
    assert sum(tiny_index.doc_lengths.values()) == n_tokens
    assert len(tiny_index) == len(counts)
    assert {t: sum(p.tf for p in pl) for t, pl in tiny_index.items()} == dict(counts)
