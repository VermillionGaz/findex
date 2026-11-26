import json
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.snippets import fetch_texts, make_snippet

FILLER = "lorem ipsum dolor sit amet. " * 20


def test_highlights_every_occurrence_in_the_window():
    s = make_snippet("The Solar System and the solar wind.", ["solar", "system"])
    assert s == "The **Solar** **System** and the **solar** wind."


def test_window_is_around_the_match_and_marked_with_ellipses():
    text = FILLER + "Here is the volcano eruption we want. " + FILLER
    s = make_snippet(text, ["volcano"], radius=40)
    assert "**volcano**" in s
    assert s.startswith("…") and s.endswith("…")
    assert len(s) < 120                         # ~2 * radius, widened to whole words


def test_prefers_the_spot_with_the_most_different_terms():
    text = "solar panels. " + FILLER + "solar wind, solar flare, solar storm. " + FILLER + \
           "the solar system formed. " + FILLER
    s = make_snippet(text, ["solar", "system"], radius=30)
    assert "**solar** **system**" in s


def test_no_match_shows_the_beginning():
    s = make_snippet("Short text about cats. " + FILLER, ["dogs"], radius=20)
    assert s.startswith("Short text about cats.") and s.endswith("…") and "**" not in s


def test_does_not_cut_words_in_half():
    s = make_snippet("abcdefghij " * 30 + "target " + "klmnopqrst " * 30, ["target"], radius=25)
    for piece in s.strip("…").split():
        assert piece in ("abcdefghij", "**target**", "klmnopqrst")


def test_matches_follow_the_tokenizer():
    # casefolding, curly apostrophes and NFC are handled like in the index
    assert make_snippet("Don’t STRASSE café", ["don't", "strasse", "café"]) == \
        "**Don’t** **STRASSE** **café**"          # shown as written, matched as normalized


def test_custom_markers():
    assert make_snippet("a cat sat", ["cat"], mark=("<b>", "</b>")) == "a <b>cat</b> sat"


def test_whitespace_is_collapsed():
    assert make_snippet("cat\n\n\tdog", ["dog"]) == "cat **dog**"


def test_fetch_texts_from_a_txt_folder(tmp_path):
    (tmp_path / "a.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second document", encoding="utf-8")
    idx = build_index(iter_documents(tmp_path))
    assert fetch_texts(idx, [1]) == {1: "second document"}


def test_fetch_texts_from_jsonl(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps({"id": i, "title": f"T{i}", "text": f"text {i}"})
                              for i in range(10)), encoding="utf-8")
    idx = build_index(iter_documents(path))
    assert fetch_texts(idx, [7, 2]) == {7: "text 7", 2: "text 2"}


def test_fetch_texts_skips_a_missing_or_changed_corpus(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"id": 1, "title": "T", "text": "hello"}), encoding="utf-8")
    idx = build_index(iter_documents(path))
    path.write_text(json.dumps({"id": 99, "title": "T", "text": "changed"}), encoding="utf-8")
    assert fetch_texts(idx, [0]) == {}          # different document now: no wrong snippet
    path.unlink()
    assert fetch_texts(idx, [0]) == {}
