"""`python -m findex ...` works the same as the installed `findex` command."""

from findex.cli import app

app(prog_name="findex")
