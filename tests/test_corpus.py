import json
import types
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents


def test_txt_directory(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("привіт", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("nope", encoding="utf-8")

    docs = list(iter_documents(tmp_path))

    assert [d.doc_id for d in docs] == ["a.txt", str(Path("sub") / "b.txt")]
    assert docs[1].text == "привіт"
    assert all(isinstance(d, Document) for d in docs)


def test_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text(
        json.dumps({"id": "x1", "text": "first"}) + "\n"
        "\n"  # blank line
        "{not json}\n"  # malformed JSON
        + json.dumps({"id": "x2"})
        + "\n"  # missing 'text'
        + json.dumps({"text": "no id"})
        + "\n",  # id falls back to line number (5)
        encoding="utf-8",
    )
    docs = list(iter_documents(p))
    assert [(d.doc_id, d.text) for d in docs] == [("x1", "first"), ("5", "no id")]


def test_bad_encoding_is_skipped_and_logged(tmp_path, caplog):
    (tmp_path / "good.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe broken")
    with caplog.at_level("WARNING"):
        docs = list(iter_documents(tmp_path))
    assert [d.doc_id for d in docs] == ["good.txt"]
    assert "bad.txt" in caplog.text


def test_missing_path_error_surfaces_late(tmp_path):
    gen = iter_documents(tmp_path / "does-not-exist")
    assert isinstance(gen, types.GeneratorType)  # calling it did NOT raise...
    with pytest.raises(FileNotFoundError):  # ...the error appears on first next()
        next(gen)


def test_document_is_immutable(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    doc = next(iter_documents(tmp_path))
    with pytest.raises(AttributeError):
        doc.text = "changed"
