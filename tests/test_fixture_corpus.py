"""End-to-end checks on the shared fixture corpus from conftest.py."""

import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.rank import BM25, TfIdf
from findex.search import Searcher, evaluate
from findex.store import load, save


def titles(index: Index, ids) -> list[str]:
    return [index.doc_meta[d].title for d in ids]


def test_every_document_is_indexed(tiny_index: Index, corpus: dict[str, str]) -> None:
    assert tiny_index.num_docs == len(corpus)
    assert [m.title for m in tiny_index.doc_meta.values()] == list(corpus)


def test_txt_folder_and_jsonl_give_the_same_postings(corpus_dir: Path, tiny_index: Index) -> None:
    from_txt = build_index(iter_documents(corpus_dir))
    assert from_txt.postings == tiny_index.postings
    assert from_txt.doc_lengths == tiny_index.doc_lengths


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("planet", ["Solar System", "Earth", "Volcano"]),
        ("programming language", ["Python", "Java"]),
        ("programming language NOT island", ["Python"]),
        ('"event loop"', ["Event loop"]),
        ('"solar system" sun', ["Solar System", "Sun"]),
        ("київ OR volcano", ["Volcano", "Київ"]),
        ("CAFÉ", ["Київ"]),
        ("python OR java NOT island", ["Python", "Event loop"]),
    ],
)
def test_known_answers(tiny_index: Index, engine, query: str, expected: list[str]) -> None:
    assert titles(tiny_index, evaluate(tiny_index, query, engine)) == expected


def test_ranking_puts_the_obvious_article_first(tiny_index: Index) -> None:
    searcher = Searcher(tiny_index)
    for scorer in (BM25(), TfIdf()):
        assert searcher.search("solar system", scorer)[0].title == "Solar System"
        assert searcher.search("volcano", scorer)[0].title == "Volcano"


def test_save_load_round_trip_of_the_fixture_index(tiny_index: Index, tmp_path: Path) -> None:
    save(tiny_index, tmp_path / "tiny.bin")
    assert load(tmp_path / "tiny.bin") == tiny_index


@pytest.mark.slow
def test_cli_pipeline_end_to_end(corpus_jsonl: Path, tmp_path: Path) -> None:
    """Build, save, and search through `python -m findex` in fresh Python processes:
    the closest thing to a user running the installed tool."""
    import os
    import subprocess
    import sys

    # Child processes on Windows write piped output in the legacy code page unless told
    # otherwise; UTF-8 everywhere keeps titles like "Київ" intact.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = tmp_path / "index.bin"
    python_m = [sys.executable, "-m"]
    subprocess.run(
        [*python_m, "findex", "index", str(corpus_jsonl), "--out", str(out)],
        check=True,
        capture_output=True,
        env=env,
    )
    result = subprocess.run(
        [*python_m, "findex", "search", str(out), "programming language NOT island", "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert [json.loads(line)["title"] for line in result.stdout.splitlines()] == ["Python"]
