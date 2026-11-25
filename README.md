
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
