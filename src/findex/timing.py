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

log = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator: log the elapsed time of every call to fn, even if it raises.

    `@timed` above `def f` means `f = timed(f)`: timed receives the original
    function and returns `wrapper`, which replaces it. `wrapper` is a closure: it
    keeps a reference to `fn` from the scope it was created in.

    The PEP 695 type parameters keep the signature visible to type checkers:
    **P stands for "whatever parameters fn takes" and R for "whatever it returns",
    so a call to the decorated function is still checked like a call to fn.
    """

    @functools.wraps(fn)  # copy __name__, __doc__, __module__, __wrapped__ onto wrapper
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            # `finally` runs whether fn returned or raised, so failures are timed too
            log.debug("%s took %.1f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper


def setup_logging(verbose: bool = False) -> None:
    """Logging for the `python -m findex.X` research scripts, called once at each one's
    entry point: status messages (INFO) by default, plus timings (DEBUG) with -v.
    Everything goes to stderr, leaving stdout for the script's actual output.
    (The installed `findex` command configures its own logging in cli.py.)"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
