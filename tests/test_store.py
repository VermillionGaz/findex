import json
import pickle
import struct
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
    with pytest.raises(TypeError, match="contains a dict, not an Index"):
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


def test_a_corpus_folder_is_not_an_index(tmp_path):
    assert detect_format(tmp_path) is None


# --- Lab 3 M1: open_index context manager -------------------------------------------------

from findex.store import open_index


@pytest.mark.parametrize("fmt", FORMATS)
def test_open_index_yields_the_index(idx, tmp_path, fmt):
    path = tmp_path / "idx"
    save(idx, path, fmt)
    with open_index(path, trust_pickle=True) as ix:
        assert ix == idx and "cat" in ix


def test_open_index_closes_the_file_even_on_exception(idx, tmp_path, monkeypatch):
    path = tmp_path / "idx"
    save(idx, path)
    opened = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        f = real_open(self, *args, **kwargs)
        opened.append(f)
        return f

    monkeypatch.setattr(Path, "open", spy)
    with pytest.raises(RuntimeError), open_index(path):
        raise RuntimeError("boom inside the with-block")
    assert opened and all(f.closed for f in opened)


def test_open_index_closes_the_file_when_loading_fails(tmp_path, monkeypatch):
    path = tmp_path / "junk"
    path.write_bytes(b"not an index at all")
    opened = []
    real_open = Path.open
    monkeypatch.setattr(
        Path, "open", lambda self, *a, **k: opened.append(real_open(self, *a, **k)) or opened[-1]
    )
    with pytest.raises(ValueError), open_index(path):
        pass
    assert opened and all(f.closed for f in opened)


# --- Lab 3 M3: positions in the binary format ----------------------------------------------


def test_binary_roundtrip_keeps_positions(idx, tmp_path):
    assert idx.has_positions
    save(idx, tmp_path / "idx")
    back = load(tmp_path / "idx")
    assert back == idx and back.has_positions


def test_binary_roundtrip_without_positions(tmp_path):
    from findex.corpus import Document

    no_pos = build_index(
        [Document(doc_id="a", path=Path("a"), text="cat dog cat", title="A")], positions=False
    )
    save(no_pos, tmp_path / "idx")
    back = load(tmp_path / "idx")
    assert back == no_pos and not back.has_positions


def test_version_1_files_still_load(idx, tmp_path, monkeypatch):
    """A file written before positions existed must keep working (without phrases)."""
    from findex import store
    from findex.index import Index, Posting

    no_pos = Index(
        {t: [Posting(p.doc_id, p.tf) for p in pl] for t, pl in idx.items()},
        idx.doc_lengths,
        idx.doc_meta,
    )
    monkeypatch.setattr(store, "VERSION", 1)  # write the old header version
    save(no_pos, tmp_path / "old")
    monkeypatch.undo()
    assert load(tmp_path / "old") == no_pos


# --- Lab 4 M2: corrupted and invalid files fail with a clear error -------------------------


def _saved(idx, tmp_path):
    path = tmp_path / "idx.bin"
    save(idx, path)
    return path


def test_truncated_file_is_reported(idx, tmp_path):
    path = _saved(idx, tmp_path)
    raw = path.read_bytes()
    for cut in (4, 10):  # ends between two numbers / in the middle of one
        path.write_bytes(raw[:-cut])  # e.g. a copy or download that stopped early
        with pytest.raises(ValueError, match="truncated"):
            load(path)


def test_unknown_version_is_reported(idx, tmp_path):
    path = _saved(idx, tmp_path)
    raw = path.read_bytes()
    # same length on purpose: the header's byte length is stored in front of it
    path.write_bytes(raw.replace(b'"version": 2', b'"version": 9', 1))
    with pytest.raises(ValueError, match="unsupported index version: 9"):
        load(path)


def test_corrupt_posting_counts_are_reported(idx, tmp_path):
    path = _saved(idx, tmp_path)
    raw = path.read_bytes()
    n = json.loads(raw[16 : 16 + struct.unpack("<Q", raw[8:16])[0]])["n_postings"]
    path.write_bytes(
        raw.replace(f'"n_postings": {n}'.encode(), f'"n_postings": {n - 1}'.encode(), 1)
    )
    with pytest.raises(ValueError, match="truncated|do not add up"):
        load(path)


def test_cannot_save_an_index_with_gaps_in_doc_ids(idx, tmp_path):
    from findex.index import Index

    gappy = Index(idx.postings, {0: 1, 5: 1}, {0: idx.doc_meta[0], 5: idx.doc_meta[0]})
    with pytest.raises(ValueError, match="0..n-1"):
        save(gappy, tmp_path / "x.bin")
