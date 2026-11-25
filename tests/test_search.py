import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.search import ENGINES, evaluate


def make_index(*texts):
    return build_index(
        Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=t, title=f"T{i}")
        for i, t in enumerate(texts)
    )


@pytest.fixture(params=["merge", "set"])
def engine(request):
    return request.param


@pytest.fixture
def idx():
    return make_index(
        "cat dog",        # 0
        "cat bird",       # 1
        "dog bird fish",  # 2
        "fish",           # 3
        "cat dog fish",   # 4
    )


def test_single_term(idx, engine):
    assert evaluate(idx, "cat", engine) == [0, 1, 4]


def test_implicit_and(idx, engine):
    assert evaluate(idx, "cat dog", engine) == [0, 4]
    assert evaluate(idx, "cat AND dog", engine) == [0, 4]


def test_or(idx, engine):
    assert evaluate(idx, "bird OR fish", engine) == [1, 2, 3, 4]


def test_not(idx, engine):
    assert evaluate(idx, "cat NOT dog", engine) == [1]
    assert evaluate(idx, "NOT cat", engine) == [2, 3]


def test_left_to_right(idx, engine):
    # ((cat AND dog) OR fish) = [0,4] | [2,3,4]
    assert evaluate(idx, "cat dog OR fish", engine) == [0, 2, 3, 4]


def test_unknown_and_case_normalization(idx, engine):
    assert evaluate(idx, "zzz", engine) == []
    assert evaluate(idx, "CAT", engine) == evaluate(idx, "cat", engine)
    assert evaluate(idx, "cat zzz", engine) == []


def test_punctuation_only_word_is_ignored(idx, engine):
    assert evaluate(idx, "cat !!!", engine) == [0, 1, 4]
    assert evaluate(idx, "", engine) == []


@pytest.mark.parametrize("bad", ["cat OR", "OR cat", "cat AND OR dog", "cat OR NOT dog", "NOT"])
def test_bad_queries(idx, bad):
    with pytest.raises(ValueError):
        evaluate(idx, bad)


def test_merge_and_set_agree_on_random_lists():
    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        for op in ("AND", "OR", "NOT"):
            assert ENGINES["merge"][op](a, b) == ENGINES["set"][op](a, b)