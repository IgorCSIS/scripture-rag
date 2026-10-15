# Vector index choice

Benchmark: `bench/bench_vectordb.py` (server side) and `bench/bench_browser_ann.mjs` (browser module under Node).
Data: synthetic unit vectors, d = 384 (the e5-small width), with planted clusters, 200 queries, k = 10.
Ground truth is exact inner-product search. Machine: the shared dev box (CPU only). Numbers are from one run;
treat them as relative, not absolute. Raw results: `docs/benchmarks/vectordb.json`, `docs/benchmarks/browser_ann.json`.

## Server side

| Backend | n | recall@10 | p50 ms | p95 ms | build s | memory MB |
|---|---|---|---|---|---|---|
| numpy brute force | 1e3 | 1.000 | 0.03 | 0.05 | 0.00 | 0.3 |
| faiss Flat | 1e3 | 1.000 | 0.03 | 0.04 | 0.05 | 14.3 |
| faiss HNSW | 1e3 | 1.000 | 0.03 | 0.04 | 0.07 | 0.8 |
| faiss IVF | 1e3 | 0.671 | 0.01 | 0.02 | 0.01 | 4.0 |
| LanceDB (IVF-PQ) | 1e3 | 1.000 | 3.39 | 4.39 | 1.34 | 160.7 |
| Chroma (HNSW) | 1e3 | 1.000 | 0.64 | 0.95 | 0.80 | 64.4 |
| numpy brute force | 1e4 | 1.000 | 0.08 | 0.27 | 0.00 | 0.0 |
| faiss Flat | 1e4 | 1.000 | 0.23 | 1.44 | 0.00 | 0.0 |
| faiss HNSW | 1e4 | 0.950 | 0.07 | 0.10 | 0.21 | 0.0 |
| faiss IVF | 1e4 | 0.929 | 0.05 | 0.07 | 0.26 | 18.4 |
| LanceDB (IVF-PQ) | 1e4 | 0.841 | 5.98 | 7.73 | 2.20 | 173.8 |
| Chroma (HNSW) | 1e4 | 0.982 | 0.79 | 1.31 | 0.77 | 69.9 |
| numpy brute force | 1e5 | 1.000 | 1.77 | 7.62 | 0.05 | 153.6 |
| faiss Flat | 1e5 | 1.000 | 2.36 | 10.78 | 0.20 | 153.6 |
| faiss HNSW | 1e5 | 0.836 | 0.14 | 0.26 | 2.34 | 153.6 |
| faiss IVF | 1e5 | 1.000 | 0.45 | 0.58 | 3.66 | 126.6 |
| LanceDB (IVF-PQ) | 1e5 | 0.669 | 6.10 | 7.99 | 21.13 | 477.5 |
| Chroma (HNSW) | 1e5 | 0.874 | 1.07 | 1.53 | 9.17 | 234.5 |
| numpy brute force | 1e6 | 1.000 | 28.11 | 49.91 | 0.46 | 1536.0 |
| faiss Flat | 1e6 | 1.000 | 53.01 | 84.06 | 1.15 | 1536.0 |
| faiss HNSW | 1e6 | 0.468 | 0.24 | 0.41 | 41.38 | 1796.2 |
| faiss IVF | 1e6 | 1.000 | 4.42 | 5.12 | 50.32 | 1772.2 |
| LanceDB (IVF-PQ) | 1e6 | 0.756 | 7.38 | 9.48 | 49.97 | 284.6 |

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
* Above about 1e5 to 1e6 chunks, use faiss IVF (exact recall in this run at 4 ms p50 for 1e6) or HNSW with a
  larger efSearch. FAISS is a single wheel with no service to run.
* LanceDB is the pick if memory matters more than latency (295 MB at 1e6 versus about 1.5 to 1.8 GB) or if a
  persistent on-disk table is wanted. Chroma adds a service-like layer with no win at this scale.

## Browser

Maintained JS/WASM options were checked: `hnswlib-wasm` (npm 0.8.2) has had no release since July 2023, so it is not
used. `usearch` is maintained (2.x, updated October 2026) but would add a WASM download and a CSP change. The site
ships a small dependency-free HNSW (`site/js/ann.js`, M 16, efConstruction 100, efSearch 64).
`core.vectorSearch` switches to it automatically above `ANN_THRESHOLD` = 20,000 vectors; below that it stays exact.

| n | brute p50 ms | HNSW p50 ms | HNSW build s | recall@10 |
|---|---|---|---|---|
| 1,000 | 0.95 | 0.64 | 1.2 | 1.000 |
| 5,000 | 3.01 | 0.26 | 4.5 | 0.850 |
| 20,000 | 13.42 | 0.69 | 18.6 | 0.794 |
| 50,000 | 47.26 | 1.33 | 70.0 | 0.942 |

Graph: `docs/benchmarks/browser_ann.png`. The build is incremental and lazy (first query after a large import),
and its cost is paid once per session. Recall below 1 is acceptable here because the 50 dense candidates are fused
with BM25 and reranked; exact search is kept for every library under the threshold.
