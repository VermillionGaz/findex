"""@timed: log how long a function call took.

    @timed
    def build_index(...): ...

    # with logging at DEBUG (CLI: -v), every call logs:
    # DEBUG findex.timing: build_index took 812.4 ms
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

log = logging.getLogger(__name__)

P = ParamSpec("P")   # "whatever parameters fn takes"...
R = TypeVar("R")     # ..."and whatever it returns": the wrapper keeps both, for type checkers


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator: log the elapsed time of every call to fn, even if it raises.

    `@timed` above `def f` means `f = timed(f)`: timed receives the original
    function and returns `wrapper`, which replaces it. `wrapper` is a closure: it
    keeps a reference to `fn` from the scope it was created in.
    """

    @functools.wraps(fn)   # copy __name__, __doc__, __module__, __wrapped__ onto wrapper
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            # `finally` runs whether fn returned or raised, so failures are timed too
            log.debug("%s took %.1f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper


def setup_logging(verbose: bool) -> None:
    """CLI helper: -v shows the timing log (DEBUG); otherwise only warnings."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
