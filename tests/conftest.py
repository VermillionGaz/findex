"""Shared fixtures. pytest loads this file automatically; every test in tests/ can use
these fixtures just by naming them as parameters -- no import needed."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents
from findex.index import Index, build_index
from findex.search import Engine

# A tiny corpus with known answers: titles -> text. Small enough to reason about by
# hand, varied enough for phrases, NOT, ranking and Unicode.
CORPUS: dict[str, str] = {
    "Solar System": "The Solar System is the Sun and the planets that orbit it. "
    "Earth is a planet in the solar system.",
    "Sun": "The Sun is a star at the centre of the solar system.",
    "Earth": "Earth is the third planet from the Sun and the only planet known to have life.",
    "Volcano": "A volcano is an opening in the crust of a planet. Lava erupts from a volcano.",
    "Python": "Python is a programming language. Python code is easy to read.",
    "Java": "Java is a programming language and also an island in Indonesia.",
    "Event loop": "An event loop waits for events and runs callbacks. Python asyncio has an event loop.",
    "Київ": "Київ is the capital of Ukraine. Café culture in Київ is famous.",
}


@pytest.fixture(scope="session")
def corpus() -> dict[str, str]:
    """The fixture corpus itself (title -> text), for tests that need the raw data."""
    return CORPUS


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The corpus as a folder of .txt files, written once per test session."""
    root = tmp_path_factory.mktemp("corpus_txt")
    for i, (title, text) in enumerate(CORPUS.items()):
        # numbered names keep the sorted file order == CORPUS order
        (root / f"{i:02d} {title}.txt").write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def corpus_jsonl(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same corpus as one .jsonl file (the Simple Wikipedia format)."""
    path = tmp_path_factory.mktemp("corpus_jsonl") / "tiny.jsonl"
    lines = [
        json.dumps({"id": i, "title": title, "text": text}, ensure_ascii=False)
        for i, (title, text) in enumerate(CORPUS.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_index(corpus_jsonl: Path) -> Index:
    """Built once per session and shared by many tests, so: never mutate it in a test.
    (Tests that need to change an index build their own with make_index.)"""
    return build_index(iter_documents(corpus_jsonl))


@pytest.fixture
def make_index() -> Callable[..., Index]:
    """Factory fixture: make_index("text one", "text two") -> a fresh Index, titled T0, T1..."""

    def make(*texts: str, positions: bool = True) -> Index:
        docs = (Document(f"d{i}", Path(f"d{i}.txt"), t, f"T{i}") for i, t in enumerate(texts))
        return build_index(docs, positions=positions)

    return make


@pytest.fixture(params=["merge", "set"])
def engine(request: pytest.FixtureRequest) -> Engine:
    """Run a test once per Boolean engine: both must give identical answers."""
    return request.param
