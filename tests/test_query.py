import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, ParseError, Phrase, Term, iter_nodes, lex, parse
from findex.search import ENGINES

T = Term  # short alias for building expected trees


# --- operators build trees ---------------------------------------------------------------
def test_operators_build_nodes():
    assert T("a") & T("b") == And(T("a"), T("b"))
    assert T("a") | T("b") == Or(T("a"), T("b"))
    assert ~T("a") == Not(T("a"))


def test_nodes_are_hashable_values():
    assert len({T("a") & T("b"), T("a") & T("b"), T("a") | T("b")}) == 2
    assert T("a") & T("b") != T("b") & T("a")          # order matters in the tree


# --- tree equality: parse(string) == tree built in code ---------------------------------
@pytest.mark.parametrize("query, tree", [
    ("cat", T("cat")),
    ("Cats", T("cats")),                                  # normalized like the corpus
    ("cat dog", T("cat") & T("dog")),                     # implicit AND
    ("cat AND dog", T("cat") & T("dog")),
    ("cat OR dog", T("cat") | T("dog")),
    ("NOT cat", ~T("cat")),
    ("NOT NOT cat", ~~T("cat")),
    ("cat NOT dog", T("cat") & ~T("dog")),
    ("a OR b c", T("a") | (T("b") & T("c"))),             # AND binds tighter than OR
    ("a b OR c", (T("a") & T("b")) | T("c")),
    ("(a OR b) c", (T("a") | T("b")) & T("c")),           # parentheses override
    ("NOT a b", ~T("a") & T("b")),                        # NOT binds tightest
    ("NOT (a b)", ~(T("a") & T("b"))),
    ("a OR b OR c", (T("a") | T("b")) | T("c")),          # left-associative
    ("a b c", (T("a") & T("b")) & T("c")),
    ("((a))", T("a")),
    ('"event loop"', Phrase(("event", "loop"))),
    ('"Event  Loop"', Phrase(("event", "loop"))),
    ('"python"', T("python")),                            # one-word phrase is a term
    ("U.S.", Phrase(("u", "s"))),                         # split word: must be adjacent
    ("e-mail", T("e-mail")),                              # your tokenizer keeps hyphens
    ("cats & dogs", T("cats") & T("dogs")),               # punctuation-only word dropped
    ('python AND (async OR await) NOT java "event loop"',
     T("python") & (T("async") | T("await")) & ~T("java") & Phrase(("event", "loop"))),
])
def test_parse_builds_the_expected_tree(query, tree):
    assert parse(query) == tree


def test_operators_must_be_uppercase():
    assert parse("cat and dog") == T("cat") & T("and") & T("dog")


@pytest.mark.parametrize("bad, where", [
    ("", "empty"),
    ("cat AND", "end of the query"),
    ("OR cat", "position 0"),
    ("cat AND OR dog", "position 8"),
    ("(cat dog", "expected ')'"),
    ("cat dog)", "position 7"),
    ('"unclosed', "unclosed quote at position 0"),
    ("()", "position 1"),
])
def test_parse_errors_say_where(bad, where):
    with pytest.raises(ParseError, match=where.replace("(", r"\(").replace(")", r"\)")):
        parse(bad)


def test_lexer():
    kinds = [t.kind for t in lex('a AND ("b c" OR NOT d)')]
    assert kinds == ["WORD", "AND", "(", "PHRASE", "OR", "NOT", "WORD", ")"]


def test_positive_terms_skip_negations():
    tree = parse('python (async OR await) NOT java "event loop"')
    assert tree.positive_terms() == ["python", "async", "await", "event", "loop"]


def test_str_is_readable():
    assert str(parse('a OR b NOT "c d"')) == '(a OR (b AND NOT "c d"))'


def test_iter_nodes_walks_the_whole_tree():
    assert sum(isinstance(n, Term) for n in iter_nodes(parse("a (b OR NOT c)"))) == 3


# --- evaluation --------------------------------------------------------------------------
TEXTS = [
    "the event loop runs async tasks",       # 0
    "an event in the loop",                  # 1  event, loop -- but not adjacent
    "python async await",                    # 2
    "java event loop",                       # 3
    "loop event",                            # 4  wrong order
    "python event loop event loop",          # 5
]


@pytest.fixture
def idx():
    return build_index(Document(f"d{i}", Path(f"d{i}"), t, f"T{i}") for i, t in enumerate(TEXTS))


@pytest.fixture(params=["merge", "set"])
def ops(request):
    return ENGINES[request.param]


def test_phrase_needs_adjacent_words_in_order(idx, ops):
    assert parse('"event loop"').evaluate(idx, ops) == [0, 3, 5]
    assert parse("event loop").evaluate(idx, ops) == [0, 1, 3, 4, 5]   # no quotes: anywhere
    assert parse('"loop event"').evaluate(idx, ops) == [4, 5]


def test_three_word_phrase(idx, ops):
    assert parse('"python async await"').evaluate(idx, ops) == [2]
    assert parse('"python await"').evaluate(idx, ops) == []


def test_phrase_with_unknown_word(idx, ops):
    assert parse('"event zebra"').evaluate(idx, ops) == []


def test_the_lab_example(idx, ops):
    q = 'python AND (async OR await) NOT java "event loop"'
    assert parse(q).evaluate(idx, ops) == []          # doc 2 has no phrase, doc 5 no async
    assert parse('(python OR java) "event loop" NOT java').evaluate(idx, ops) == [5]


def test_not_alone_is_everything_else(idx, ops):
    assert parse("NOT event").evaluate(idx, ops) == [2]


def test_phrase_on_an_index_without_positions_fails_clearly():
    no_pos = build_index([Document("a", Path("a"), "event loop", "A")], positions=False)
    with pytest.raises(ValueError, match="positions"):
        parse('"event loop"').evaluate(no_pos, ENGINES["merge"])


def brute_force(node, docs: list[list[str]]) -> set[int]:
    """Reference semantics, written as directly as possible."""
    if isinstance(node, Term):
        return {i for i, d in enumerate(docs) if node.text in d}
    if isinstance(node, Phrase):
        n = len(node.terms)
        return {i for i, d in enumerate(docs)
                if any(tuple(d[j:j + n]) == node.terms for j in range(len(d)))}
    if isinstance(node, And):
        return brute_force(node.left, docs) & brute_force(node.right, docs)
    if isinstance(node, Or):
        return brute_force(node.left, docs) | brute_force(node.right, docs)
    return set(range(len(docs))) - brute_force(node.child, docs)


def random_tree(rng, depth=0):
    words = ["a", "b", "c", "d"]
    r = rng.random()
    if depth > 3 or r < 0.3:
        return T(rng.choice(words))
    if r < 0.4:
        return Phrase((rng.choice(words), rng.choice(words)))
    if r < 0.55:
        return ~random_tree(rng, depth + 1)
    left, right = random_tree(rng, depth + 1), random_tree(rng, depth + 1)
    return left & right if r < 0.8 else left | right


def test_evaluation_matches_brute_force_on_random_queries(ops):
    rng = random.Random(42)
    docs = [[rng.choice("abcd") for _ in range(rng.randint(0, 8))] for _ in range(40)]
    idx = build_index(Document(f"d{i}", Path("x"), " ".join(d), "") for i, d in enumerate(docs))
    for _ in range(500):
        tree = random_tree(rng)
        assert tree.evaluate(idx, ops) == sorted(brute_force(tree, docs)), str(tree)


def test_parsing_str_of_a_tree_gives_the_same_tree():
    rng = random.Random(7)
    for _ in range(300):
        tree = random_tree(rng)
        assert parse(str(tree)) == tree, str(tree)
