
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

---
# Sanity check: 

(findex) PS G:\python projects\findex> uv run python -m findex.sanity index.bin
<!-- Index(terms=909,375, docs=241,787, avg_doc_length=175.9) -->

### Check 1 — a rare term outranks a common one

Query `the OR margarines`. `the` is in 218,158 of 241,787 documents; `margarines` is in 1.

| Scorer | Doc A: only the rare term (tf=1) | Doc B: only the common term (tf=1) | Result |
|---|---|---|---|
| BM25 | 15.168 (Margarine, 94 tokens) | 0.130 (Macromolecule, 94 tokens) | V holds |
| TF-IDF | 12.396 (Margarine, 94 tokens) | 0.103 (Macromolecule, 94 tokens) | V holds |

Why: idf. A term in few documents says much more about the ones it's in.

### Check 2 — BM25 saturates, TF-IDF doesn't

Term `98058` (df=1) in document “Distance” (176 tokens; average is 176).

| tf | BM25 score | BM25 gain per extra occurrence | TF-IDF score | TF-IDF gain per extra occurrence |
|---|---|---|---|---|
| 1 | 11.986 | +11.9865 | 12.396 | +12.3958 |
| 2 | 17.125 | +5.1386 | 24.792 | +12.3958 |
| 3 | 19.980 | +2.8552 | 37.187 | +12.3958 |
| 5 | 23.056 | +1.5376 | 61.979 | +12.3958 |
| 10 | 26.064 | +0.6017 | 123.958 | +12.3958 |
| 19 | 27.781 | +0.1908 | 235.520 | +12.3958 |
| 20 | 27.883 | +0.1021 | 247.916 | +12.3958 |

With BM25 the 20th occurrence adds 0.9% of what the 1st added: V holds. With TF-IDF it adds 100% — score grows linearly forever.

### Check 3 — a short document beats a long one with the same single match

Query `the`; it occurs exactly once in both documents.

| Scorer | Short: “Nitta Shrine” (5 tokens) | Long: “List of minor planets: 1001–2000” (18978 tokens) | Result |
|---|---|---|---|
| BM25 | 0.183 | 0.002 | V holds |
| TF-IDF | 0.103 | 0.103 | tie — TF-IDF ignores length |

The long document is 3796× longer: one mention in it is weaker evidence the document is *about* `the`. BM25's `b` parameter encodes that.

<!-- 3/3 checks hold -->