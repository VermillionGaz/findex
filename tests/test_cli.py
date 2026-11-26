"""The `findex` command, tested in-process with typer's CliRunner: it invokes the app
like a shell would and captures stdout, stderr and the exit code separately."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index
from findex.store import save

runner = CliRunner()


@pytest.fixture(scope="session")
def index_file(tiny_index: Index, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "tiny.bin"
    save(tiny_index, path)
    return path


def run(*args: object):
    # A wide virtual terminal, so rich doesn't wrap table cells mid-phrase.
    return runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})


def plain(output: str) -> str:
    """Table text without borders or alignment: compare what is shown, not how it's drawn."""
    for ch in "│┃┏┓┗┛┡┩┠┨━─┌┐└┘├┤┬┴┼╇╈╋┳┻":
        output = output.replace(ch, " ")
    return " ".join(output.split())


# --- findex (root) --------------------------------------------------------------------------
def test_help_lists_the_three_commands() -> None:
    result = run("--help")
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_no_arguments_shows_help() -> None:
    result = run()
    assert "Usage" in result.stdout + result.stderr


# --- findex index ---------------------------------------------------------------------------
def test_index_builds_and_saves(corpus_jsonl: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.bin"
    result = run("index", corpus_jsonl, "--out", out)
    assert result.exit_code == 0, result.stderr
    assert out.exists()
    assert "indexed 8 documents" in result.stderr
    assert result.stdout == ""  # nothing but results ever goes to stdout


def test_index_limit_and_txt_folder(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.bin"
    assert run("index", corpus_dir, "-o", out, "--limit", "3").exit_code == 0
    assert "documents 3 " in plain(run("stats", out).stdout)


def test_index_missing_corpus_is_one_clean_error(tmp_path: Path) -> None:
    result = run("index", tmp_path / "nope.jsonl", "--out", tmp_path / "x.bin")
    assert result.exit_code == 1
    assert result.stderr.strip().splitlines() == [
        f"error: corpus not found: {tmp_path / 'nope.jsonl'}"
    ]


def test_index_empty_corpus_is_an_error(tmp_path: Path) -> None:
    result = run("index", tmp_path, "--out", tmp_path / "x.bin")  # a folder with no .txt files
    assert result.exit_code == 1 and "no documents" in result.stderr


# --- findex search --------------------------------------------------------------------------
def test_search_prints_a_ranked_table_with_snippets(index_file: Path) -> None:
    result = run("search", index_file, "programming language NOT island")
    assert result.exit_code == 0, result.stderr
    text = plain(result.stdout)
    assert "1 matches" in text
    assert "Python" in text and "Java" not in text
    assert "easy to read" in text  # the snippet


def test_search_json_lines(index_file: Path) -> None:
    result = run("search", index_file, "planet", "--json", "--k", "2")
    assert result.exit_code == 0
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["rank"] for r in records] == [1, 2]
    assert set(records[0]) == {"rank", "score", "doc_id", "id", "title", "snippet"}
    assert records[0]["score"] >= records[1]["score"]
    assert "planet" in records[0]["snippet"].lower()


def test_json_stdout_stays_clean_even_with_debug_logging(index_file: Path) -> None:
    result = run("-vv", "search", index_file, "volcano", "--json")
    assert result.exit_code == 0
    assert [json.loads(line)["title"] for line in result.stdout.splitlines()] == ["Volcano"]
    assert "cache miss" in result.stderr  # the logs went to stderr


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_search_scorers(index_file: Path, scorer: str) -> None:
    result = run("search", index_file, "solar system", "--scorer", scorer, "--json")
    assert json.loads(result.stdout.splitlines()[0])["title"] == "Solar System"


def test_search_phrase_and_unicode(index_file: Path) -> None:
    result = run("search", index_file, '"event loop" OR київ', "--json", "--no-snippets")
    titles = {json.loads(line)["title"] for line in result.stdout.splitlines()}
    assert titles == {"Event loop", "Київ"}
    assert all("snippet" not in json.loads(line) for line in result.stdout.splitlines())


def test_bad_query_fails_before_loading_the_index(tmp_path: Path) -> None:
    # The index file doesn't even exist: the query error must come first.
    result = run("search", tmp_path / "missing.bin", "cat AND")
    assert result.exit_code == 2
    assert result.stderr.startswith("error: expected a term")
    assert len(result.stderr.strip().splitlines()) == 1


@pytest.mark.parametrize(
    ("make_file", "message"),
    [
        (lambda p: None, "index not found"),
        (lambda p: p.write_text("hello"), "is not a findex index file"),
    ],
)
def test_search_bad_index_file(tmp_path: Path, make_file, message: str) -> None:
    path = tmp_path / "idx.bin"
    make_file(path)
    result = run("search", path, "cat")
    assert result.exit_code == 1 and message in result.stderr


def test_pickle_needs_trust(tiny_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "idx.pkl"
    save(tiny_index, path, "pickle")
    refused = run("search", path, "volcano")
    assert refused.exit_code == 1 and "pickle" in refused.stderr
    assert run("search", path, "volcano", "--trust-pickle").exit_code == 0


def test_phrase_on_index_without_positions(corpus_jsonl: Path, tmp_path: Path) -> None:
    out = tmp_path / "nopos.bin"
    run("index", corpus_jsonl, "-o", out, "--no-positions")
    result = run("search", out, '"event loop"')
    assert result.exit_code == 1 and "positions" in result.stderr


def test_invalid_option_values_are_rejected(index_file: Path) -> None:
    assert run("search", index_file, "cat", "--k", "0").exit_code == 2
    assert run("search", index_file, "cat", "--scorer", "magic").exit_code == 2


# --- findex stats ---------------------------------------------------------------------------
def test_stats(index_file: Path) -> None:
    result = run("stats", index_file, "--top", "3")
    assert result.exit_code == 0
    text = plain(result.stdout)
    assert "documents 8 " in text and "top 3 terms" in text
    assert "term occurrences documents % of tokens the " in text  # "the" is the top term


# --- logging levels ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("flags", "expect_info", "expect_debug"),
    [
        ([], False, False),
        (["-v"], True, False),
        (["-vv"], True, True),
    ],
)
def test_verbosity_levels(
    index_file: Path, flags: list[str], expect_info: bool, expect_debug: bool
) -> None:
    result = run(*flags, "search", index_file, "volcano", "--json")
    assert ("documents match" in result.stderr) == expect_info
    assert ("took" in result.stderr) == expect_debug


@pytest.mark.parametrize(
    "flags",
    [
        ["--executor", "serial"],
        ["--executor", "threads", "--workers", "3"],
        pytest.param(["--executor", "processes", "--workers", "2"], marks=pytest.mark.slow),
    ],
)
def test_index_executors_write_identical_files(
    corpus_jsonl: Path, tmp_path: Path, flags: list[str]
) -> None:
    reference, out = tmp_path / "ref.bin", tmp_path / "out.bin"
    assert run("index", corpus_jsonl, "-o", reference).exit_code == 0
    result = run("index", corpus_jsonl, "-o", out, *flags)
    assert result.exit_code == 0, result.stderr
    assert out.read_bytes() == reference.read_bytes()


def test_index_rejects_zero_workers(corpus_jsonl: Path, tmp_path: Path) -> None:
    assert run("index", corpus_jsonl, "-o", tmp_path / "x", "--workers", "0").exit_code == 2
