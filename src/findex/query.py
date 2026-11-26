"""The query language: a tree of small objects, and a parser that builds it.

    parse('python AND (async OR await) NOT java "event loop"')
    == Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(("event", "loop"))

Grammar (operators UPPERCASE; precedence NOT > AND > OR; a space means AND):

    or_expr  := and_expr ('OR' and_expr)*
    and_expr := not_expr (['AND'] not_expr)*
    not_expr := 'NOT' not_expr | atom
    atom     := WORD | "QUOTED PHRASE" | '(' or_expr ')'

Each grammar rule is one method of the parser, and each calls the rules below
it -- "recursive descent". Lower-precedence rules sit higher up, so they end up
nearer the root of the tree and are applied last.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from .index import Index
from .tokenize import tokenize

DocIds = list[int]  # sorted ascending, no duplicates
type OpName = Literal["AND", "OR", "NOT"]
type Ops = Mapping[OpName, Callable[[DocIds, DocIds], DocIds]]  # operator -> merge function


class ParseError(ValueError):
    pass


# =============================================================================================
# Query nodes
# =============================================================================================
class Query:
    """Base for every node: gives the whole tree the &, | and ~ operators.

    Python turns `a & b` into a.__and__(b), `a | b` into a.__or__(b) and `~a`
    into a.__invert__(). So queries can be built in code, no string needed:
        Term("cat") & ~Term("dog")   ->   And(Term("cat"), Not(Term("dog")))
    """

    __slots__ = ()

    def __and__(self, other: Query) -> Query:
        return And(self, other)

    def __or__(self, other: Query) -> Query:
        return Or(self, other)

    def __invert__(self) -> Query:
        return Not(self)

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        """The sorted ids of every document matching this (sub)query."""
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Terms that should contribute to the ranking score (not negated ones)."""
        raise NotImplementedError


# frozen=True: nodes are values, so dataclass gives them __eq__ (and __hash__)
# comparing fields. Two separately built trees with the same shape are ==, which
# is exactly what "tree-equality tests" need.


@dataclass(frozen=True, slots=True)
class Term(Query):
    text: str  # one normalized token

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        return [p.doc_id for p in index.get(self.text, [])]

    def positive_terms(self) -> list[str]:
        return [self.text]

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Phrase(Query):
    terms: tuple[str, ...]  # normalized tokens, in order

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        if not index.has_positions:
            raise ValueError(
                "phrase queries need positions: rebuild the index without --no-positions"
            )
        # 1. Documents containing ALL the words (cheap: postings merge).
        plists = [index.get(t, []) for t in self.terms]
        docs = [p.doc_id for p in plists[0]]
        for plist in plists[1:]:
            docs = ops["AND"](docs, [p.doc_id for p in plist])
        # 2. Of those, the ones where the words are adjacent and in order.
        wanted = set(docs)
        where = [{p.doc_id: p.positions for p in plist if p.doc_id in wanted} for plist in plists]
        return [d for d in docs if _adjacent(*(w[d] for w in where))]

    def positive_terms(self) -> list[str]:
        return list(self.terms)

    def __str__(self) -> str:
        return '"' + " ".join(self.terms) + '"'


def _adjacent(*positions: tuple[int, ...]) -> bool:
    """Is there a start p with p in positions[0], p+1 in positions[1], p+2 in ...?

    Shift each word's positions back by its place in the phrase; a phrase
    occurrence is a start offset that survives every shift. Intersect as we go
    and stop early once nothing is left.
    """
    starts = set(positions[0])
    for i, pos in enumerate(positions[1:], start=1):
        starts &= {p - i for p in pos}
        if not starts:
            return False
    return True


@dataclass(frozen=True, slots=True)
class And(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        # "a NOT b" is And(a, Not(b)): compute it as a minus b directly instead of
        # building "every document except b" first.
        if isinstance(self.right, Not):
            return ops["NOT"](self.left.evaluate(index, ops), self.right.child.evaluate(index, ops))
        if isinstance(self.left, Not):
            return ops["NOT"](self.right.evaluate(index, ops), self.left.child.evaluate(index, ops))
        return ops["AND"](self.left.evaluate(index, ops), self.right.evaluate(index, ops))

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True, slots=True)
class Or(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        return ops["OR"](self.left.evaluate(index, ops), self.right.evaluate(index, ops))

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True, slots=True)
class Not(Query):
    child: Query

    def evaluate(self, index: Index, ops: Ops) -> DocIds:
        # On its own (e.g. the query "NOT java"), NOT means "every other document".
        return ops["NOT"](list(range(index.num_docs)), self.child.evaluate(index, ops))

    def positive_terms(self) -> list[str]:
        return []  # what you exclude doesn't score

    def __str__(self) -> str:
        return f"NOT {self.child}"


# =============================================================================================
# Lexer: query string -> tokens
# =============================================================================================
@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # "WORD", "PHRASE", "AND", "OR", "NOT", "(", ")"
    text: str
    pos: int  # character offset in the query, for error messages


_LEX = re.compile(r'\s*(?:(?P<phrase>"[^"]*"?)|(?P<paren>[()])|(?P<word>[^\s()"]+))')


def lex(query: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(query):
        m = _LEX.match(query, pos)
        if m is None or m.lastgroup is None:  # only trailing whitespace left
            break
        start, pos = m.start(m.lastgroup), m.end()
        if m.lastgroup == "phrase":
            text = m.group("phrase")
            if len(text) < 2 or not text.endswith('"'):
                raise ParseError(f"unclosed quote at position {start}")
            if _words(text[1:-1]) is not None:  # drop empty phrases like ""
                tokens.append(Token("PHRASE", text[1:-1], start))
        elif m.lastgroup == "paren":
            tokens.append(Token(m.group("paren"), m.group("paren"), start))
        else:
            word = m.group("word")
            if word in ("AND", "OR", "NOT"):
                tokens.append(Token(word, word, start))
            elif _words(word) is not None:  # "cats & dogs": the "&" is dropped,
                tokens.append(Token("WORD", word, start))  # like Lab 2 did
    return tokens


# =============================================================================================
# Parser: tokens -> tree
# =============================================================================================
class _Parser:
    def __init__(self, query: str) -> None:
        self.query = query
        self.tokens = lex(query)
        self.i = 0

    # --- helpers ---------------------------------------------------------------------------
    def peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def error(self, message: str) -> ParseError:
        tok = self.peek()
        where = f"at position {tok.pos} ({tok.text!r})" if tok else "at the end of the query"
        return ParseError(f"{message} {where}: {self.query!r}")

    def starts_operand(self) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind in ("WORD", "PHRASE", "(", "NOT")

    # --- one method per grammar rule ---------------------------------------------------------
    def parse(self) -> Query:
        if not self.tokens:
            raise ParseError(f"empty query (no searchable words): {self.query!r}")
        tree = self.or_expr()
        if self.peek() is not None:  # e.g. a stray ")"
            raise self.error("unexpected token")
        return tree

    def or_expr(self) -> Query:
        node = self.and_expr()
        while (tok := self.peek()) is not None and tok.kind == "OR":
            self.take()
            node = node | self.and_expr()  # left-associative: ((a | b) | c)
        return node

    def and_expr(self) -> Query:
        node = self.not_expr()
        while True:
            tok = self.peek()
            if tok is not None and tok.kind == "AND":
                self.take()
            elif not self.starts_operand():  # implicit AND only if an operand follows
                return node
            node = node & self.not_expr()

    def not_expr(self) -> Query:
        tok = self.peek()
        if tok is not None and tok.kind == "NOT":
            self.take()
            return ~self.not_expr()
        return self.atom()

    def atom(self) -> Query:
        tok = self.peek()
        if tok is None:
            raise self.error("expected a term")
        if tok.kind == "(":
            self.take()
            node = self.or_expr()
            if (close := self.peek()) is None or close.kind != ")":
                raise self.error("expected ')'")
            self.take()
            return node
        if tok.kind in ("WORD", "PHRASE"):
            self.take()
            node = _words(tok.text)
            assert node is not None  # the lexer only keeps words with tokens
            return node
        raise self.error("expected a term")


def _words(text: str) -> Query | None:
    """Normalize text exactly like the corpus was: one token -> Term, several -> Phrase.

    So `cats` -> Term("cats"), `"Event Loop"` -> Phrase(("event", "loop")), and a
    bare `U.S.` -> Phrase(("u", "s")) -- the pieces must appear together.
    """
    tokens = tuple(tokenize(text))
    if not tokens:
        return None
    return Term(tokens[0]) if len(tokens) == 1 else Phrase(tokens)


def parse(query: str) -> Query:
    """Parse a query string into a tree. Raises ParseError with the position on bad input."""
    return _Parser(query).parse()


def iter_nodes(node: Query) -> Iterator[Query]:
    """Every node of the tree, depth-first (handy for tests and debugging)."""
    yield node
    for child in (
        getattr(node, name) for name in ("left", "right", "child") if hasattr(node, name)
    ):
        yield from iter_nodes(child)
