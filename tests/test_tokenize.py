import types

import pytest

from findex.tokenize import token_spans, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", [], id="empty"),
        pytest.param("   \n\t ", [], id="whitespace-only"),
        pytest.param("Hello WORLD", ["hello", "world"], id="case"),
        pytest.param("Hi, there! (ok?)", ["hi", "there", "ok"], id="punctuation"),
        pytest.param("Привіт, Київ!", ["привіт", "київ"], id="cyrillic"),
        pytest.param("Ελληνικά ΣΟΦΙΑ", ["ελληνικά", "σοφια"], id="greek"),
        pytest.param("東京 tokyo", ["東京", "tokyo"], id="cjk"),
        pytest.param("Straße", ["strasse"], id="casefold-eszett"),
        pytest.param("cafe\u0301", ["caf\u00e9"], id="combining-mark-to-nfc"),
        pytest.param("caf\u00e9", ["caf\u00e9"], id="precomposed"),
        pytest.param("don't rock-n-roll", ["don't", "rock-n-roll"], id="apostrophe-hyphen"),
        pytest.param("don’t", ["don't"], id="curly-apostrophe"),
        pytest.param("'quoted' ''double''", ["quoted", "double"], id="quotes-stripped"),
        pytest.param(
            "python3 2026 snake_case", ["python3", "2026", "snake", "case"], id="digits-underscore"
        ),
        pytest.param("U.S. 3.14", ["u", "s", "3", "14"], id="abbreviations-and-decimals"),
        pytest.param("100,000", ["100", "000"], id="thousands-separator"),
        pytest.param("-leading trailing-", ["leading", "trailing"], id="edge-hyphens"),
        pytest.param("emoji 🎉 here", ["emoji", "here"], id="emoji-dropped"),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


@pytest.mark.parametrize(
    ("a", "b"), [("don't", "don’t"), ("cafe\u0301", "caf\u00e9"), ("STRASSE", "Straße")]
)
def test_spellings_that_must_match(a: str, b: str) -> None:
    assert list(tokenize(a)) == list(tokenize(b))


def test_is_lazy() -> None:
    assert isinstance(tokenize("x"), types.GeneratorType)


def test_token_spans_agree_with_tokenize_and_point_at_the_words() -> None:
    text = "Don’t panic: the U.S. Straße, café and Київ, 3.14!"
    spans = list(token_spans(text))
    assert [t for t, _, _ in spans] == list(tokenize(text))
    assert text[spans[0][1] : spans[0][2]] == "Don’t"
