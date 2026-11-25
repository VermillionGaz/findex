
| Version | Document | Peak memory | Time Elapsed |
|---------|----------|-------------|--------------|
| Eager   | 20000    | 512.1 Mb    | 20.50 s      |
| Lazy    | 20000    | 20.3 Mb     | 23.51 s      |

Lazy consumes significantly less memory, while taking ~15% longer.
this is likely due to pausing of generators for every token.
the times are inflated because of tracemalloc as well, but that is non-factor here.
