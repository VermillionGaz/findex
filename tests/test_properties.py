"""Property-based tests: instead of hand-picked examples, Hypothesis generates hundreds
of inputs per test and checks that a rule (a "property") always holds. When one fails,
it shrinks the input to the smallest example that still fails.

Run with --hypothesis-show-statistics to see how many examples each test tried.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import Index, Posting, build_index
from findex.query import And, Not, Or, Phrase, Query, Term, parse
from findex.rank import BM25, TfIdf, score_docs, top_k
from findex.search import ENGINES, Engine
from findex.store import load, save
from findex.tokenize import tokenize

pytestmark = pytest.mark.slow  # every test in this file runs hundreds of examples

# --- strategies: recipes for generating inputs ------------------------------------------

# Sorted, duplicate-free doc id lists -- the shape every merge function expects.
doc_ids = st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted)

# Text from a deliberately nasty alphabet: a few words (so terms repeat and phrases
# occur), plus Unicode letters, combining marks, apostrophes, digits and punctuation.
WORDS = ["cat", "dog", "the", "Київ", "café", "don't", "U.S.", "3.14", "a-b"]
texts = st.lists(
    st.one_of(st.sampled_from(WORDS), st.text(max_size=6)),
    max_size=12,
).map(" ".join)

corpora = st.lists(texts, min_size=1, max_size=8)

# Small vocabulary of already-normalized terms for query trees.
TERMS = st.sampled_from(["cat", "dog", "bird", "fish"])


def query_trees() -> st.SearchStrategy[Query]:
    """Random query trees, built recursively from Term/Phrase leaves."""
    leaves = st.one_of(
        TERMS.map(Term),
        st.lists(TERMS, min_size=2, max_size=3).map(lambda ts: Phrase(tuple(ts))),
    )
    return st.recursive(
        leaves,
        lambda children: st.one_of(
            st.builds(And, children, children),
            st.builds(Or, children, children),
            st.builds(Not, children),
        ),
        max_leaves=8,
    )


def build(texts_: list[str], positions: bool = True) -> Index:
    docs = (Document(f"d{i}", Path(f"d{i}"), t, f"T{i}") for i, t in enumerate(texts_))
    return build_index(docs, positions=positions)


# --- 1. oracle: the merge algorithms agree with Python's sets -----------------------------
@given(doc_ids, doc_ids)
def test_merges_agree_with_set_operations(a: list[int], b: list[int]) -> None:
    merge = ENGINES["merge"]
    assert merge["AND"](a, b) == sorted(set(a) & set(b))
    assert merge["OR"](a, b) == sorted(set(a) | set(b))
    assert merge["NOT"](a, b) == sorted(set(a) - set(b))


# --- 2. idempotence: tokenizing tokens changes nothing --------------------------------------
@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


# --- 3. round-trip: load(save(index)) == index ----------------------------------------------
@given(corpora, st.booleans())
@settings(max_examples=50, deadline=None)  # file I/O: fewer examples, no per-example deadline
def test_save_load_round_trip(texts_: list[str], positions: bool) -> None:
    index = build(texts_, positions)
    # tmp_path can't be used with @given (one directory would be shared by every
    # generated example), so each example makes its own temporary directory.
    with tempfile.TemporaryDirectory() as tmp:
        save(index, Path(tmp) / "index.bin")
        assert load(Path(tmp) / "index.bin") == index


# --- 4. invariants: what must hold for ANY index ---------------------------------------------
@given(corpora)
def test_index_invariants(texts_: list[str]) -> None:
    index = build(texts_)
    tokens_of = {i: list(tokenize(t)) for i, t in enumerate(texts_)}
    for term, plist in index.items():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids))  # sorted, no duplicates
        for p in plist:
            assert p.tf == len(p.positions) >= 1  # tf counts the positions
            assert all(tokens_of[p.doc_id][pos] == term for pos in p.positions)
    for doc_id, tokens in tokens_of.items():  # lengths add up
        assert index.doc_length(doc_id) == len(tokens)
        assert Counter(tokens) == Counter(
            {t: p.tf for t, pl in index.items() for p in pl if p.doc_id == doc_id}
        )


# --- 5. the parser: printing a tree and parsing it back gives the same tree -------------------
@given(query_trees())
def test_parse_str_round_trip(tree: Query) -> None:
    assert parse(str(tree)) == tree


# --- 6. oracle: query evaluation agrees with brute force ---------------------------------------
def brute_force(node: Query, docs: list[list[str]]) -> set[int]:
    if isinstance(node, Term):
        return {i for i, d in enumerate(docs) if node.text in d}
    if isinstance(node, Phrase):
        n = len(node.terms)
        return {
            i
            for i, d in enumerate(docs)
            if any(tuple(d[j : j + n]) == node.terms for j in range(len(d)))
        }
    if isinstance(node, And):
        return brute_force(node.left, docs) & brute_force(node.right, docs)
    if isinstance(node, Or):
        return brute_force(node.left, docs) | brute_force(node.right, docs)
    assert isinstance(node, Not)
    return set(range(len(docs))) - brute_force(node.child, docs)


@given(
    st.lists(st.lists(st.sampled_from(["cat", "dog", "bird", "fish"]), max_size=8), min_size=1),
    query_trees(),
    st.sampled_from(["merge", "set"]),
)
# The case Hypothesis found: an index whose documents are all empty has no postings,
# and a phrase query on it raised "needs positions" instead of returning [].
# @example pins it so it is checked on every run, not just when Hypothesis rediscovers it.
@example(docs=[[]], tree=Phrase(("cat", "cat")), engine="merge")
def test_evaluation_agrees_with_brute_force(
    docs: list[list[str]], tree: Query, engine: Engine
) -> None:
    index = build([" ".join(d) for d in docs])
    assert tree.evaluate(index, ENGINES[engine]) == sorted(brute_force(tree, docs))


# --- 7. ranking: scores are sane and top_k is a correct partial sort ---------------------------
@given(corpora, st.lists(st.sampled_from(WORDS), min_size=1, max_size=3), st.integers(1, 10))
def test_ranking_properties(texts_: list[str], query_words: list[str], k: int) -> None:
    index = build(texts_)
    terms = [t for w in query_words for t in tokenize(w)]
    for scorer in (BM25(), TfIdf()):
        scores = score_docs(index, terms, scorer)
        assert all(s >= 0 for s in scores.values())  # never negative
        best = top_k(scores, k)
        assert len(best) == min(k, len(scores))
        assert [s for _, s in best] == sorted((s for _, s in best), reverse=True)
        if best:  # nothing left out beats them
            assert min(s for _, s in best) >= max(
                (s for d, s in scores.items() if d not in dict(best)), default=0.0
            )


# --- 8. BM25 saturates: each extra occurrence adds something, but less than the last ----------
@given(
    corpora,
    st.floats(min_value=0.1, max_value=3.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.integers(min_value=1, max_value=50),
)
def test_bm25_is_increasing_with_diminishing_returns(
    texts_: list[str], k1: float, b: float, tf: int
) -> None:
    index = build(texts_)
    if not index:
        return  # every document was empty: no term to score
    term = next(iter(index))
    doc = index[term][0].doc_id
    scorer = BM25(k1=k1, b=b)
    s0, s1, s2 = (scorer(term, Posting(doc, n), index) for n in (tf - 1, tf, tf + 1))
    assert s0 < s1 < s2  # more occurrences always score higher...
    assert s2 - s1 < s1 - s0  # ...but each one adds less than the previous
