from pathlib import Path

from findex.corpus import Document
from findex.index import DocMeta, Index, Posting, build_index


def make(*texts):
    return [Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=t, title=f"T{i}")
            for i, t in enumerate(texts)]


def test_postings_are_sorted_and_counted():
    idx = build_index(make("cat dog cat", "dog", "cat bird"))
    assert idx.get("cat") == [Posting(0, 2), Posting(2, 1)]
    assert idx.get("dog") == [Posting(0, 1), Posting(1, 1)]
    ids = [p.doc_id for p in idx.get("cat")]
    assert ids == sorted(ids)


def test_doc_lengths_and_meta():
    idx = build_index(make("a b c", "a"))
    assert idx.doc_lengths == {0: 3, 1: 1}
    assert idx.doc_meta[1] == DocMeta(ext_id="d1", path="d1.txt", title="T1")
    assert idx.n_docs == 2


def test_unknown_term_does_not_pollute_index():
    idx = build_index(make("a"))
    assert idx.get("zzz") == []
    assert "zzz" not in idx.postings
    assert type(idx.postings) is dict  # not a defaultdict


def test_accepts_a_generator_and_empty_input():
    assert build_index(iter([])).n_docs == 0
    assert build_index(d for d in make("x y")).get("x") == [Posting(0, 1)]


def test_posting_and_meta_are_hashable_and_slotted():
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert not hasattr(Posting(1, 2), "__dict__")
    assert hash(DocMeta("a", "b", "c")) == hash(DocMeta("a", "b", "c"))


def test_matches_term_counts_with_tokenizer_rules():
    idx = build_index(make("Don't don’t DON'T"))
    assert idx.get("don't") == [Posting(0, 3)]