import pickle
from array import array
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.memstudy import (
    VARIANTS,
    PlainPosting,
    SlotPosting,
    as_pairs,
    build_arrays,
    build_objects,
    count_postings,
    measure,
)

TEXTS = ["apple banana apple", "banana cherry", "Apple APPLE apple cherry"]
EXPECTED = {
    "apple": [(0, 2), (2, 3)],
    "banana": [(0, 1), (1, 1)],
    "cherry": [(1, 1), (2, 1)],
}


def docs():
    # String ids, exactly like the real corpus: internal int ids come from enumerate.
    return (Document(doc_id=f"ext{i}", path=Path(f"d{i}.txt"), text=t) for i, t in enumerate(TEXTS))


@pytest.mark.parametrize("name", VARIANTS)
def test_each_variant_builds_the_expected_postings(name):
    _, build = VARIANTS[name]
    assert as_pairs(build(docs())) == EXPECTED


def test_variants_match_the_real_index():
    real = build_index(docs())
    assert as_pairs(real.postings) == as_pairs(build_arrays(docs()))


def test_postings_are_sorted_by_doc_id():
    for plist in as_pairs(build_arrays(docs())).values():
        ids = [d for d, _ in plist]
        assert ids == sorted(ids)


def test_array_variant_stores_raw_4_byte_ints():
    doc_ids, tfs = build_arrays(docs())["apple"]
    assert isinstance(doc_ids, array) and doc_ids.typecode == "I" and doc_ids.itemsize == 4
    assert list(doc_ids) == [0, 2] and list(tfs) == [2, 3]


def test_slots_variant_has_no_dict_and_plain_does():
    assert not hasattr(SlotPosting(1, 1), "__dict__")
    assert hasattr(PlainPosting(1, 1), "__dict__")


@pytest.mark.parametrize("cls", [PlainPosting, SlotPosting])
def test_postings_are_frozen_and_hashable(cls):
    p = cls(1, 2)
    assert {p, cls(1, 2)} == {p}
    with pytest.raises(FrozenInstanceError):
        p.tf = 5


@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_survives_a_pickle_roundtrip(name):
    _, build = VARIANTS[name]
    assert as_pairs(pickle.loads(pickle.dumps(build(docs())))) == EXPECTED


@pytest.mark.parametrize("name", VARIANTS)
def test_measure_reports_sane_numbers_and_cleans_up(name, tmp_path):
    r = measure(name, docs(), tmp_path)
    assert r["n_postings"] == count_postings(r["postings"]) == 6
    assert r["n_terms"] == 3
    assert r["peak"] >= r["retained"] > 0
    assert r["size"] > 0 and r["load_s"] >= 0
    assert as_pairs(r["postings"]) == EXPECTED
    assert list(tmp_path.iterdir()) == [], "temporary pickle was not deleted"
