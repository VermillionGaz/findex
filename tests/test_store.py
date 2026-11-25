import pickle
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.search import evaluate
from findex.store import FORMATS, MAGIC, detect_format, load, save


@pytest.fixture
def idx():
    texts = ["cat dog", "Київ café cat", "", "dog bird fish", "naïve 2026 rock-n-roll"]
    return build_index(
        Document(doc_id=f"id{i}", path=Path(f"p{i}.txt"), text=t, title=f"Title ü {i}")
        for i, t in enumerate(texts)
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_is_exact(idx, tmp_path, fmt):
    path = tmp_path / "idx"
    save(idx, path, fmt)
    assert detect_format(path) == fmt
    assert load(path, trust_pickle=True) == idx


@pytest.mark.parametrize("fmt", FORMATS)
def test_searches_give_same_answers_after_reload(idx, tmp_path, fmt):
    path = tmp_path / "idx"
    save(idx, path, fmt)
    again = load(path, trust_pickle=True)
    for q in ["cat", "cat dog", "dog OR bird", "cat NOT dog", "київ"]:
        assert evaluate(again, q) == evaluate(idx, q)


@pytest.mark.parametrize("fmt", FORMATS)
def test_empty_index(tmp_path, fmt):
    empty = build_index([])
    path = tmp_path / "idx"
    save(empty, path, fmt)
    assert load(path, trust_pickle=True) == empty


def test_pickle_is_refused_without_trust(idx, tmp_path):
    path = tmp_path / "idx"
    save(idx, path, "pickle")
    with pytest.raises(PermissionError):
        load(path)


def test_binary_needs_no_trust(idx, tmp_path):
    path = tmp_path / "idx"
    save(idx, path, "binary")
    assert load(path) == idx


def test_not_an_index_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello")
    assert detect_format(path) is None
    with pytest.raises(ValueError):
        load(path)


def test_truncated_binary_is_detected(idx, tmp_path):
    path = tmp_path / "idx"
    save(idx, path, "binary")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 7])
    with pytest.raises(ValueError):
        load(path)


def test_wrong_object_in_pickle_is_rejected(tmp_path):
    path = tmp_path / "idx"
    path.write_bytes(pickle.dumps({"not": "an index"}, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(ValueError):
        load(path, trust_pickle=True)


def test_unknown_format_name(idx, tmp_path):
    with pytest.raises(ValueError):
        save(idx, tmp_path / "idx", "xml")


def test_binary_file_starts_with_magic(idx, tmp_path):
    path = tmp_path / "idx"
    save(idx, path, "binary")
    assert path.read_bytes().startswith(MAGIC)


def test_pickle_saved_by_the_cli_loads_elsewhere(tmp_path):
    """Regression: classes must be pickled as findex.index.X, not __main__.X."""
    import subprocess
    import sys
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id": "1", "title": "T", "text": "cat dog"}\n', encoding="utf-8")
    out = tmp_path / "idx.pkl"
    cmd = [sys.executable, "-m", "findex.index", str(corpus)]
    cmd += ["--out", str(out), "--format", "pickle"]
    subprocess.run(cmd, check=True, capture_output=True)
    assert load(out, trust_pickle=True).get("cat")[0].doc_id == 0