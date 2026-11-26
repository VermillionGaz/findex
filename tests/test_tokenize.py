from findex.tokenize import tokenize


def toks(s):
    return list(tokenize(s))


def test_empty():
    assert toks("") == []


def test_case():
    assert toks("Hello WORLD") == ["hello", "world"]


def test_punctuation():
    assert toks("Hi, there! (ok?)") == ["hi", "there", "ok"]


def test_cyrillic():
    assert toks("Привіт, Київ!") == ["привіт", "київ"]


def test_combining_mark_equals_precomposed():
    assert toks("cafe\u0301") == toks("caf\u00e9") == ["caf\u00e9"]


def test_casefold_eszett():
    assert toks("Straße") == ["strasse"]


def test_apostrophe_hyphen():
    assert toks("don't rock-n-roll") == ["don't", "rock-n-roll"]


def test_digits_underscore():
    assert toks("python3 2026 snake_case") == ["python3", "2026", "snake", "case"]


def test_is_lazy():
    import types

    assert isinstance(tokenize("x"), types.GeneratorType)


def test_curly_and_straight_apostrophe_match():
    assert toks("don't") == toks("don’t") == ["don't"]


def test_quotes_are_stripped():
    assert toks("'quoted' ''double''") == ["quoted", "double"]


def test_numbers_and_abbreviations_split():
    assert toks("U.S. 3.14") == ["u", "s", "3", "14"]


def test_token_spans_agree_with_tokenize_and_point_at_the_words():
    from findex.tokenize import token_spans

    text = "Don’t panic: the U.S. Straße, café and Київ, 3.14!"
    spans = list(token_spans(text))
    assert [t for t, _, _ in spans] == list(tokenize(text))
    assert text[spans[0][1] : spans[0][2]] == "Don’t"
