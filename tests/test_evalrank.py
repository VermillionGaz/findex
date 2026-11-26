import itertools
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.evalrank import (
    label, pool, precision_at_k, read_judgments, read_queries, report, write_judgments,
)
from findex.index import build_index
from findex.search import Searcher


@pytest.fixture
def idx():
    texts = ["cat"] * 3 + ["cat cat dog"] * 3 + ["dog"] * 4
    return build_index(Document(f"d{i}", Path(f"d{i}"), t, f"T{i}") for i, t in enumerate(texts))


def test_precision_at_5():
    judged = {"a": True, "b": False, "c": True}
    assert precision_at_k(["a", "b", "c", "d", "e"], judged) == (0.4, 2)   # d, e unjudged
    assert precision_at_k(["a"], judged) == (0.2, 0)                       # fewer than 5: /5
    assert precision_at_k([], judged) == (0.0, 0)


def test_read_queries_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text('# comment\n\ncat dog\n  "solar system"  \n', encoding="utf-8")
    assert read_queries(f) == ["cat dog", '"solar system"']


def test_judgments_roundtrip(tmp_path):
    j = {"cat": {"Café": True, "T2": False}}
    write_judgments(tmp_path / "sub" / "j.json", j)
    assert read_judgments(tmp_path / "sub" / "j.json") == j
    assert read_judgments(tmp_path / "missing.json") == {}


def test_pool_is_the_union_of_both_top_5s_without_duplicates(idx):
    p = pool(Searcher(idx), "cat")
    assert len(p) == len(set(p)) <= 10 and set(p) <= set(range(6))


def test_label_saves_after_each_answer_and_resumes(idx, tmp_path):
    path = tmp_path / "j.json"
    answers = iter(["huh?", "y", "n", "q"])     # an invalid answer is asked again
    label(idx, ["cat"], path, ask=lambda _: next(answers))
    assert list(read_judgments(path)["cat"].values()) == [True, False]

    asked = []
    label(idx, ["cat"], path, ask=lambda prompt: asked.append(prompt) or "y")
    final = read_judgments(path)["cat"]
    assert len(final) == len(pool(Searcher(idx), "cat"))
    assert len(asked) == len(final) - 2          # the two judged ones weren't asked again


def test_report_table(idx, tmp_path):
    searcher = Searcher(idx)
    judged = {r.title: r.title in ("T3", "T4", "T5") for r in searcher.search("cat", k=10)}
    table = report(idx, ["cat"], {"cat": judged})
    assert table.splitlines()[0] == "| # | Query | Matches | TF-IDF P@5 | BM25 P@5 |"
    assert "| 1 | `cat` | 6 |" in table and "**mean**" in table
    assert "unjudged" not in table
