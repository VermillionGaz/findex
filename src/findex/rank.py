"""Relevance ranking: a Scorer protocol with TF-IDF and BM25 implementations.

A scorer answers one question: "how much does this one occurrence record
(term, posting) say this document is about the query?" Search adds those
answers up per document and keeps the top k.

    tf       how often the term occurs in the document      (posting.tf)
    df       how many documents contain the term            (index.df(term))
    N        how many documents there are                   (index.num_docs)
    idf      how rare -- and so how informative -- a term is: log(N / df)
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from operator import itemgetter
from typing import Literal, Protocol, runtime_checkable

from .index import Index, Posting


# --- the protocol -------------------------------------------------------------------------
@runtime_checkable
class Scorer(Protocol):
    """Anything callable as scorer(term, posting, index) -> float is a Scorer.

    No inheritance needed: a plain function with this signature qualifies just as
    well as an instance of a class with __call__. That's structural typing --
    "behaves like" instead of "is a".
    """

    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


# --- implementations ------------------------------------------------------------------------
@dataclass(frozen=True)
class TfIdf:
    """score = tf * log(N / df)

    Linear in tf (the 20th occurrence counts as much as the 1st) and blind to
    document length. A term in every document gets idf = log(1) = 0.
    """

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        return posting.tf * math.log(index.num_docs / df)


@dataclass(frozen=True)
class BM25:
    """Okapi BM25, the default ranking function in Lucene and Elasticsearch.

                       tf * (k1 + 1)
    score = idf * ---------------------------------------------
                   tf + k1 * (1 - b + b * doc_len / avg_doc_len)

    k1  term-frequency saturation: the score approaches idf * (k1 + 1) as tf
        grows, so extra repetitions add less and less.
    b   length normalization: 0 = ignore length, 1 = fully normalize. A match in
        a long document counts less than the same match in a short one.

    idf = log(1 + (N - df + 0.5) / (df + 0.5)) is Lucene's variant: like
    log(N / df) for rare terms, but never negative for very common ones.
    """

    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        if df == 0:
            return 0.0
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        length_ratio = index.doc_length(posting.doc_id) / index.avg_doc_length
        tf = posting.tf
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length_ratio))


type ScorerName = Literal["bm25", "tfidf"]

DEFAULT_SCORER = BM25()
SCORERS: dict[ScorerName, Scorer] = {"bm25": DEFAULT_SCORER, "tfidf": TfIdf()}


# --- results --------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class SearchResult:
    """One ranked hit.

    order=True generates <, <=, >, >= comparing the fields *in declaration order*,
    as a tuple. That's why score comes first: sorted(results) orders by score,
    and doc_id only breaks ties. Use sorted(results, reverse=True) for best-first.
    """

    score: float
    doc_id: int
    title: str


# --- scoring and top-k ----------------------------------------------------------------------
def score_docs(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer,
    candidates: set[int] | None = None,
) -> dict[int, float]:
    """Accumulate scorer(term, posting, index) per document.

    Walks each term's postings once (term-at-a-time). If `candidates` is given,
    documents outside it are skipped -- that's how a Boolean filter combines
    with ranking.
    """
    scores: dict[int, float] = {}
    for term in terms:
        for posting in index.get(term, []):
            if candidates is None or posting.doc_id in candidates:
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + scorer(
                    term, posting, index
                )
    return scores


def top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """The k highest-scoring (doc_id, score) pairs, best first.

    heapq.nlargest keeps a heap of size k while scanning: O(n log k), versus
    O(n log n) for sorting all n scores when we only want 10 of them.
    """
    return heapq.nlargest(k, scores.items(), key=itemgetter(1))


def rank(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer = DEFAULT_SCORER,
    k: int = 10,
    candidates: set[int] | None = None,
) -> list[SearchResult]:
    scores = score_docs(index, terms, scorer, candidates)
    return [
        SearchResult(score=score, doc_id=doc_id, title=index.doc_meta[doc_id].title)
        for doc_id, score in top_k(scores, k)
    ]
