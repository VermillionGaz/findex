# findex

A full-text search engine over Simple English Wikipedia, built lab by lab in the
[Python course](https://github.com/rmalkevy/Programming-Practice-Projects/blob/main/courses/python/README.md).

```
uv run python -m findex.stats data/simplewiki.jsonl --limit 20000      # corpus statistics
uv run python -m findex.index data/simplewiki.jsonl --out index.bin    # build + save the index
uv run python -m findex.search index.bin "cat NOT dog"                 # Boolean search
uv run python -m findex.search index.bin --bench                       # merge vs. set benchmark
uv run python -m findex.store data/simplewiki.jsonl --limit 20000      # format benchmark
uv run python -m findex.memstudy data/simplewiki.jsonl --limit 20000   # memory study
uv run pytest
```

---

## Lab 1 — Streams, Not Lists

The corpus is streamed one document at a time (`corpus.py`) through a lazy
tokenizer (`tokenize.py`); `stats.py` computes document count, token count,
vocabulary size and the top terms in a single pass.

| Version | Documents | Peak memory | Time elapsed |
|---|---|---|---|
| Eager | 20,000 | 512.1 MB | 20.50 s |
| Lazy | 20,000 | 20.3 MB | 23.51 s |

Lazy consumes about 25× less memory while taking ~15% longer. The extra time
is likely the cost of suspending and resuming generators for every token. Both
timings are inflated by `tracemalloc`, but equally, so the comparison holds.

---

## Lab 2 — The Inverted Index

`index.py` builds `term -> postings sorted by doc_id` in one streaming pass:
each document's terms are counted with a `Counter`, flushed into the postings,
and discarded. Documents get sequential internal ids (0, 1, 2, …), so every
postings list is sorted for free. `Posting` and `DocMeta` are
`@dataclass(frozen=True, slots=True)`: immutable, hashable, and without a
per-instance `__dict__`.

### Boolean search: merge vs. sets (M2)

`search.py` supports implicit AND, `AND`, `OR` and `NOT`, evaluated left to
right, using either a two-pointer merge (`--engine merge`) or Python sets
(`--engine set`). Benchmark from `findex.search index.bin --bench`:

<!-- paste your --bench output numbers here -->
| Case | Terms (doc freq) | Op | Merge µs | Set µs |
|---|---|---|---|---|
| 2 most common | | AND | | |
| 2 most common | | OR | | |
| 2 rarest | | AND | | |
| 2 rarest | | OR | | |
| common × rare | | AND | | |
| common × rare | | OR | | |

Both algorithms are O(len(a) + len(b)), so the difference is constant
factors. On long lists **sets win**: the set operation's loop runs in C,
while the merge runs one Python bytecode loop iteration per element. On tiny
lists **the merge wins**, because building two set objects costs more than a
few comparisons. On the **common × rare AND**, the merge can win by a lot: it
stops as soon as the shorter list is exhausted, while sets always hash every
element of the long list. How much it wins depends on where the rare term's
doc id sits: near the end of the long list, the merge walks almost all of it.

### Persistence (M3)

`store.py` saves and loads the index as pickle or as a custom binary format
(an 8-byte magic number, a JSON header with terms and document metadata, then
raw little-endian `array('I')` buffers). `load()` detects the format from the
file's first bytes.

| Format | Size (MB) | Save (s) | Load (s) | Round trip |
|---|---|---|---|---|
| pickle | 58.9 | 9.19 | 9.74 | OK |
| binary | 33.9 | 0.63 | 5.11 | OK |

The binary format is about 40% smaller and saves 15× faster, because it
writes a few large blocks of raw integers instead of one pickled object per
posting. Loading is still slow for both, since both rebuild a `Posting`
object for every posting.

**Why pickle is dangerous:** a pickle file is not data, it's a program for
the unpickler. It can name any importable function (such as `os.system`) with
arguments, and `pickle.load` will call it, before your code ever sees the
result. Loading a pickle someone else made is remote code execution. That's
why `load()` refuses pickle files unless called with `trust_pickle=True`
(`--trust-pickle` on the CLI). The binary format only parses JSON and
integers, so it's safe to load from anywhere.

### Memory study (M4)

`memstudy.py` builds the postings three ways over the same corpus and checks
that all three hold identical data.

<!-- paste the output of findex.memstudy here -->
Corpus: ____, ____ terms, ____ postings, CPython ____.

| Postings representation | Peak memory (build) | Retained after build | Bytes / posting | Index file size | Load time |
|---|---|---|---|---|---|
| `list[Posting]` with plain `@dataclass` | | | | | |
| `list[Posting]` with `slots=True` | | | | | |
| `array('I')` pairs (doc_ids, tfs) — no objects | | | | | |

**Where the bytes went.** In both object-based rows, each posting costs an
8-byte pointer in the list, a Python object for the posting itself, and a
28-byte `int` object for any `doc_id` above 256 (smaller ints, which covers
most `tf` values, are shared cached singletons and cost nothing). A plain
dataclass posting measures about 120 B in total. On CPython 3.11+ that is
*not* the classic "~100 extra bytes for a `__dict__`": attribute values are
stored inline in the object, and a real dict is only created if something
asks for `obj.__dict__`. `slots=True` removes even the inline-values area,
leaving an object header plus two fixed slots: about 88 B per posting,
roughly a quarter less. The `array('I')` row removes the objects entirely:
each posting is two raw 4-byte integers stored contiguously, 8 B, with no
pointers, headers or `int` objects. The bytes-per-posting column is higher
than these per-posting costs because it also spreads the term dict and the
term strings across all postings. The same logic explains the file sizes and
load times: pickling objects means writing, and later rebuilding, one object
per posting, while an array pickles as one block of bytes.

One surprise: the slotted dataclass loads **slower** than the plain one.
`@dataclass(frozen=True, slots=True)` generates a pure-Python `__setstate__`
(a frozen class can't be filled in by ordinary attribute assignment), so
unpickling runs Python code once per posting. The plain class takes pickle's
C fast path instead.
