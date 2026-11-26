import logging

import pytest

from findex.timing import timed


@timed
def add(a, b=1):
    """Add two numbers."""
    return a + b


def test_wraps_keeps_the_function_identity():
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."
    assert add.__wrapped__(2, 3) == 5          # the original is still reachable


def test_passes_arguments_and_return_value_through():
    assert add(2) == 3 and add(2, b=5) == 7


def test_logs_elapsed_time(caplog):
    with caplog.at_level(logging.DEBUG, logger="findex.timing"):
        add(1)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage().startswith("add took ")
    assert caplog.records[0].getMessage().endswith(" ms")


def test_still_logs_and_reraises_when_the_function_fails(caplog):
    @timed
    def boom():
        raise RuntimeError("no")

    with caplog.at_level(logging.DEBUG, logger="findex.timing"), pytest.raises(RuntimeError):
        boom()
    assert "boom took" in caplog.text


def test_methods_are_logged_with_their_class_name(caplog):
    class Thing:
        @timed
        def work(self):
            return 42

    with caplog.at_level(logging.DEBUG, logger="findex.timing"):
        assert Thing().work() == 42
    assert "Thing.work took" in caplog.text


def test_build_and_load_are_timed(caplog, tmp_path):
    from pathlib import Path

    from findex.corpus import Document
    from findex.index import build_index
    from findex.store import load, save

    with caplog.at_level(logging.DEBUG, logger="findex.timing"):
        idx = build_index([Document("a", Path("a"), "cat", "A")])
        save(idx, tmp_path / "i")
        load(tmp_path / "i")
    assert "build_index took" in caplog.text and "load took" in caplog.text
