import heapq
import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Posting, build_index
from findex.rank import BM25, SearchResult, Scorer, TfIdf, rank, score_docs, top_k
from findex.search import positive_terms, search


def make(*texts):
    return build_index(
        Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=t, title=f"T{i}")
        for i, t in enumerate(texts)
    )


FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit "


# --- the protocol ---------------------------------------------------------------------------
def tf_only(term, posting, index):
    return float(posting.tf)


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), tf_only])
def test_anything_with_the_right_call_signature_is_a_scorer(scorer):
    assert isinstance(scorer, Scorer)


def test_a_plain_function_works_as_a_scorer():
    idx = make("cat cat cat", "cat", "dog")
    assert [r.doc_id for r in rank(idx, ["cat"], tf_only)] == [0, 1]


def test_scorers_are_immutable_values():
    assert BM25() == BM25(k1=1.5, b=0.75) and BM25(k1=2.0) != BM25()
    with pytest.raises(AttributeError):
        BM25().k1 = 3


# --- the three sanity checks, on a tiny hand-made index ---------------------------------------
@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_rare_term_outranks_common_term(scorer):
    # "common" is in 4 of 5 docs, "rare" in 1. Doc 0 has only rare, doc 1 only common.
    idx = make("rare x y", "common x y", "common a b", "common c d", "common e f")
    scores = score_docs(idx, ["common", "rare"], scorer)
    assert scores[0] > scores[1]


def test_bm25_saturates_and_tfidf_does_not():
    idx = make("cat " + FILLER, FILLER, FILLER)
    bm25 = [BM25()("cat", Posting(0, tf), idx) for tf in range(21)]
    tfidf = [TfIdf()("cat", Posting(0, tf), idx) for tf in range(21)]
    first, last = bm25[1] - bm25[0], bm25[20] - bm25[19]
    assert last < 0.05 * first                                   # 20th adds almost nothing
    assert all(b2 > b1 for b1, b2 in zip(bm25, bm25[1:]))        # but never decreases
    assert tfidf[20] - tfidf[19] == pytest.approx(tfidf[1] - tfidf[0])  # linear


def test_short_doc_outranks_long_doc_with_bm25_but_ties_with_tfidf():
    idx = make("cat dog", "cat " + FILLER * 20, "dog")
    short, long = Posting(0, 1), Posting(1, 1)
    assert BM25()("cat", short, idx) > BM25()("cat", long, idx)
    assert TfIdf()("cat", short, idx) == TfIdf()("cat", long, idx)


def test_bm25_parameters_do_what_they_claim():
    idx = make("cat dog", "cat " + FILLER * 20, "dog")
    # approx: e.g. idf*9*1/9 and idf*1*1/1 can differ in the last bit of a float
    no_length = BM25(b=0)
    assert no_length("cat", Posting(0, 1), idx) == pytest.approx(no_length("cat", Posting(1, 1), idx))
    binary_tf = BM25(k1=0)
    assert binary_tf("cat", Posting(0, 1), idx) == pytest.approx(binary_tf("cat", Posting(0, 9), idx))


def test_bm25_idf_never_negative_even_for_a_term_in_every_doc():
    idx = make("the cat", "the dog", "the bird")
    assert all(BM25()("the", p, idx) > 0 for p in idx["the"])
    assert all(TfIdf()("the", p, idx) == 0 for p in idx["the"])   # log(N/N) = 0


def test_unknown_term_scores_zero_and_ranks_nothing():
    idx = make("cat")
    assert BM25()("zzz", Posting(0, 1), idx) == 0.0
    assert rank(idx, ["zzz"]) == []


# --- top-k and results ------------------------------------------------------------------------
def test_top_k_matches_a_full_sort():
    rng = random.Random(0)
    scores = {d: rng.random() for d in range(1000)}
    expected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    assert top_k(scores, 10) == expected


def test_top_k_with_k_larger_than_the_results():
    assert top_k({1: 0.5, 2: 0.9}, 10) == [(2, 0.9), (1, 0.5)]


def test_search_results_sort_by_score_then_doc_id():
    a, b, c = SearchResult(1.0, 5, "x"), SearchResult(2.0, 1, "y"), SearchResult(1.0, 3, "z")
    assert sorted([a, b, c]) == [c, a, b]
    assert sorted([a, b, c], reverse=True)[0] == b


def test_rank_returns_best_first_with_titles():
    idx = make("cat", "cat cat cat", "dog")
    results = rank(idx, ["cat"], TfIdf())
    assert [r.doc_id for r in results] == [1, 0]
    assert results[0].title == "T1"
    assert results[0].score > results[1].score


# --- search(): Boolean filter + ranking -------------------------------------------------------
def test_positive_terms_skip_operators_and_negated_terms():
    assert positive_terms("cat AND dog OR bird NOT fish") == ["cat", "dog", "bird"]
    assert positive_terms("NOT fish cat") == ["cat"]
    assert positive_terms("Cats cats") == ["cats", "cats"]


def test_search_filters_with_boolean_and_ranks_the_rest():
    idx = make("cat dog", "cat cat cat", "cat fish", "dog")
    assert [r.doc_id for r in search(idx, "cat NOT fish", TfIdf())] == [1, 0]
    assert [r.doc_id for r in search(idx, "cat dog")] == [0]          # implicit AND
    assert {r.doc_id for r in search(idx, "cat OR dog")} == {0, 1, 2, 3}


def test_search_respects_k():
    idx = make(*["cat"] * 30)
    assert len(search(idx, "cat", k=5)) == 5


def test_search_with_only_a_negation_returns_unranked_matches():
    idx = make("cat", "dog", "bird")
    results = search(idx, "NOT cat")
    assert [r.doc_id for r in results] == [1, 2] and all(r.score == 0 for r in results)


def test_sanity_script_checks_hold_on_a_small_index(capsys):
    from findex.sanity import check_rare_beats_common, check_saturation, check_short_beats_long
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(300)]
    weights = [1 / (i + 1) for i in range(300)]
    idx = make(*(" ".join(rng.choices(vocab, weights, k=rng.choice([10, 50, 400])))
                 for _ in range(200)))
    assert check_rare_beats_common(idx)
    assert check_saturation(idx)
    assert check_short_beats_long(idx)
