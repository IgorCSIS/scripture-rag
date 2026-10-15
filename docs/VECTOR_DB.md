# Vector index choice

Benchmark: `bench/bench_vectordb.py` (server side) and `bench/bench_browser_ann.mjs` (browser module under Node).
Data: synthetic unit vectors, d = 384 (the e5-small width), with planted clusters, 200 queries, k = 10.
Ground truth is exact inner-product search. Machine: the shared dev box (CPU only). Numbers are from one run;
treat them as relative, not absolute. Raw results: `docs/benchmarks/vectordb.json`, `docs/benchmarks/browser_ann.json`.

## Server side

| Backend | n | recall@10 | p50 ms | p95 ms | build s | memory MB |
|---|---|---|---|---|---|---|
| numpy brute force | 1e3 | 1.000 | 0.03 | 0.05 | 0.00 | 0.3 |
| faiss Flat | 1e3 | 1.000 | 0.04 | 0.05 | 0.04 | 14.2 |
| faiss HNSW | 1e3 | 1.000 | 0.03 | 0.03 | 0.10 | 0.8 |
| faiss IVF | 1e3 | 0.671 | 0.01 | 0.02 | 0.02 | 4.0 |
| LanceDB (IVF-PQ) | 1e3 | 1.000 | 3.32 | 4.20 | 1.31 | 159.9 |
| Chroma (HNSW) | 1e3 | 1.000 | 0.60 | 0.93 | 0.64 | 64.4 |
| numpy brute force | 1e4 | 1.000 | 0.08 | 0.29 | 0.00 | 0.0 |
| faiss Flat | 1e4 | 1.000 | 0.21 | 1.54 | 0.01 | 0.0 |
| faiss HNSW | 1e4 | 0.950 | 0.08 | 0.11 | 0.20 | 0.0 |
| faiss IVF | 1e4 | 0.929 | 0.04 | 0.08 | 0.33 | 18.4 |
| LanceDB (IVF-PQ) | 1e4 | 0.834 | 5.94 | 7.47 | 2.30 | 233.8 |
| Chroma (HNSW) | 1e4 | 0.978 | 0.82 | 1.26 | 0.76 | 69.9 |
| numpy brute force | 1e5 | 1.000 | 1.76 | 6.18 | 0.04 | 153.6 |
| faiss Flat | 1e5 | 1.000 | 3.48 | 10.20 | 0.13 | 153.6 |
| faiss HNSW | 1e5 | 0.836 | 0.13 | 0.23 | 2.29 | 159.6 |
| faiss IVF | 1e5 | 1.000 | 0.44 | 0.67 | 3.74 | 126.6 |
| LanceDB (IVF-PQ) | 1e5 | 0.658 | 6.21 | 7.61 | 20.62 | 315.5 |
| Chroma (HNSW) | 1e5 | 0.854 | 1.04 | 1.40 | 8.86 | 228.5 |
| numpy brute force | 1e6 | 1.000 | 27.03 | 33.64 | 0.41 | 1536.1 |
| faiss Flat | 1e6 | 1.000 | 49.76 | 62.77 | 2.38 | 1537.0 |
| faiss HNSW | 1e6 | 0.468 | 0.27 | 0.47 | 42.20 | 1797.3 |
| faiss IVF | 1e6 | 1.000 | 4.76 | 6.16 | 50.05 | 1756.5 |
| LanceDB (IVF-PQ) | 1e6 | 0.766 | 7.67 | 9.79 | 47.75 | 399.9 |

Chroma was capped at 1e5 (insert time). Memory is the RSS growth measured around the build, so small values are
noisy and can read 0. HNSW recall falls at 1e6 because efSearch was held at 64 for every n; raising it trades
latency for recall. IVF recall at 1e3 is low because nlist is too large for so few points. These are tuning
artifacts, not properties of the libraries.

Graphs: `docs/benchmarks/vectordb_recall_at_10.png`, `vectordb_p50_ms.png`, `vectordb_build_s.png`, `vectordb_mem_mb.png`.

Big-O: brute force O(n d) per query, no build; HNSW expected O(ef M d log n) query, O(n log n M d) build,
O(n (d + M)) memory; IVF O(n d i) build (k-means), O(nprobe (n / nlist) d + nlist d) query; IVF-PQ the same with
d replaced by the code size, plus O(refine k d) for re-scoring.

## Recommendation

* Today's libraries are a few thousand to tens of thousands of chunks (the full test corpus is 6,538 verse chunks).
  Exact numpy brute force answers in well under 5 ms at 1e4 to 1e5 with recall 1.0 and no build step, so it stays
  the default. Exact search also keeps answers reproducible, which matters for citation audits.
* Above about 1e5 to 1e6 chunks, use faiss IVF (exact recall in this run at 4.8 ms p50 for 1e6) or HNSW with a
  larger efSearch. FAISS is a single wheel with no service to run.
* LanceDB is the pick if memory matters more than latency (400 MB at 1e6 versus about 1.5 to 1.8 GB) or if a
  persistent on-disk table is wanted. Chroma adds a service-like layer with no win at this scale.

## Browser

Maintained JS/WASM options were checked: `hnswlib-wasm` (npm 0.8.2) has had no release since July 2023, so it is not
used. `usearch` is maintained (2.x, updated October 2026) but would add a WASM download and a CSP change. The site
ships a small dependency-free HNSW (`site/js/ann.js`, M 16, efConstruction 100, efSearch 64).
`core.vectorSearch` switches to it automatically above `ANN_THRESHOLD` = 20,000 vectors; below that it stays exact.

| n | brute p50 ms | HNSW p50 ms | HNSW build s | recall@10 |
|---|---|---|---|---|
| 1,000 | 0.62 | 0.39 | 0.9 | 1.000 |
| 5,000 | 3.19 | 0.28 | 3.9 | 0.850 |
| 20,000 | 15.97 | 0.82 | 20.5 | 0.794 |
| 50,000 | 43.86 | 1.31 | 73.6 | 0.942 |

Graph: `docs/benchmarks/browser_ann.png`. The build is incremental and lazy (first query after a large import),
and its cost is paid once per session. Recall below 1 is acceptable here because the 50 dense candidates are fused
with BM25 and reranked; exact search is kept for every library under the threshold.
