from pathlib import Path

from findex.corpus import Document
from findex.index import DocMeta, Posting, build_index


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
    assert idx.num_docs == 2


def test_unknown_term_does_not_pollute_index():
    idx = build_index(make("a"))
    assert idx.get("zzz") is None and idx.get("zzz", []) == []
    assert "zzz" not in idx.postings
    assert type(idx.postings) is dict  # not a defaultdict


def test_accepts_a_generator_and_empty_input():
    assert build_index(iter([])).num_docs == 0
    assert build_index(d for d in make("x y")).get("x") == [Posting(0, 1)]


def test_posting_and_meta_are_hashable_and_slotted():
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert not hasattr(Posting(1, 2), "__dict__")
    assert hash(DocMeta("a", "b", "c")) == hash(DocMeta("a", "b", "c"))


def test_matches_term_counts_with_tokenizer_rules():
    idx = build_index(make("Don't don’t DON'T"))
    assert idx.get("don't") == [Posting(0, 3)]

# --- Lab 3 M1: Index as a Mapping -------------------------------------------------------

from collections.abc import Mapping

import pytest

from findex.index import Index


def lab3_idx():
    return build_index(make("cat dog cat", "dog", "cat bird fish fish"))


def test_index_is_a_mapping_of_postings():
    idx = lab3_idx()
    assert isinstance(idx, Mapping)
    assert len(idx) == 4                                   # cat, dog, bird, fish
    assert "cat" in idx and "zzz" not in idx
    assert idx["cat"] == [Posting(0, 2), Posting(2, 1)]
    assert set(idx) == {"cat", "dog", "bird", "fish"}
    assert dict(idx.items())["dog"] == [Posting(0, 1), Posting(1, 1)]  # inherited from Mapping


def test_unknown_term_raises_keyerror_and_does_not_insert():
    idx = lab3_idx()
    with pytest.raises(KeyError):
        idx["zzz"]
    assert "zzz" not in idx and len(idx) == 4


def test_statistics():
    idx = lab3_idx()
    assert idx.num_docs == 3
    assert idx.doc_length(2) == 4
    assert idx.avg_doc_length == pytest.approx((3 + 1 + 4) / 3)
    assert idx.df("cat") == 2 and idx.df("fish") == 1 and idx.df("zzz") == 0


def test_avg_doc_length_is_computed_once():
    idx = lab3_idx()
    first = idx.avg_doc_length
    idx.doc_lengths[0] = 1000            # deliberately mutate behind its back...
    assert idx.avg_doc_length == first   # ...the cached value is not recomputed
    assert "avg_doc_length" in vars(idx)  # cached_property stored it in __dict__


def test_empty_index():
    idx = build_index([])
    assert len(idx) == 0 and idx.num_docs == 0 and idx.avg_doc_length == 0.0
    assert repr(idx) == "Index(terms=0, docs=0, avg_doc_length=0.0)"


def test_repr():
    assert repr(lab3_idx()) == "Index(terms=4, docs=3, avg_doc_length=2.7)"


def test_equality_covers_metadata_not_just_postings():
    a = lab3_idx()
    b = lab3_idx()
    assert a == b
    b.doc_meta[0] = DocMeta(ext_id="other", path="x", title="X")
    assert a != b                        # Mapping.__eq__ alone would say they're equal
    assert a != dict(a.items())          # an Index never equals a plain dict
    with pytest.raises(TypeError):
        hash(a)
