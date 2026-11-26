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


@pytest.fixture
def idx():
    return make_index(
        "cat dog",  # 0
        "cat bird",  # 1
        "dog bird fish",  # 2
        "fish",  # 3
        "cat dog fish",  # 4
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


@pytest.mark.parametrize("empty", ["", "   ", "!!!", '""'])
def test_empty_query_is_an_error(idx, empty):
    # Lab 3: the parser rejects a query with nothing to search for (Lab 2 returned []).
    with pytest.raises(ValueError):
        evaluate(idx, empty)


def test_or_not_now_works(idx, engine):
    # Lab 2's left-to-right evaluator rejected this; the parser handles it.
    everything = set(range(idx.num_docs))
    dogs = set(evaluate(idx, "dog", engine))
    cats = set(evaluate(idx, "cat", engine))
    assert set(evaluate(idx, "cat OR NOT dog", engine)) == cats | (everything - dogs)


@pytest.mark.parametrize(
    "bad", ["cat OR", "OR cat", "cat AND OR dog", "NOT", "(cat", "cat)", '"cat']
)
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


# --- Lab 3 M4: Searcher and its cache ------------------------------------------------------

import logging

from findex.search import Searcher


def test_repeated_query_hits_the_cache(idx, caplog):
    searcher = Searcher(idx)
    with caplog.at_level(logging.DEBUG, logger="findex.search"):
        first = searcher.search("cat dog")
        second = searcher.search("cat dog")
    assert first == second
    assert searcher._match.cache_info().hits == 1
    assert "cache miss" in caplog.text and "cache HIT" in caplog.text


def test_equivalent_spellings_share_one_cache_entry(idx):
    searcher = Searcher(idx)
    for q in ["cat dog", "Cat   dog", "cat AND dog", "(cat) AND (dog)"]:
        searcher.search(q)
    info = searcher._match.cache_info()
    assert (info.misses, info.hits, info.currsize) == (1, 3, 1)


def test_each_searcher_has_its_own_cache(idx):
    a, b = Searcher(idx), Searcher(idx)
    a.search("cat")
    assert b._match.cache_info().currsize == 0


def test_cached_ids_cannot_be_mutated_by_callers(idx):
    searcher = Searcher(idx)
    from findex.query import parse

    ids = searcher.matching(parse("cat"))
    assert isinstance(ids, tuple)  # a list here could be corrupted for everyone


def test_unranked_search_and_last_total(idx):
    searcher = Searcher(idx)
    results = searcher.search("cat", scorer=None, k=2)
    assert [r.score for r in results] == [0.0, 0.0]
    assert searcher.last_total == len(evaluate(idx, "cat"))
