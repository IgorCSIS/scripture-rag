# Vector index choice

Benchmark: `bench/bench_vectordb.py` (server side) and `bench/bench_browser_ann.mjs` (browser module under Node).
Data: synthetic unit vectors, d = 384 (the e5-small width), with planted clusters, 200 queries, k = 10.
Ground truth is exact inner-product search. Machine: the shared dev box (CPU only). Numbers are from one run;
treat them as relative, not absolute. Raw results: `docs/benchmarks/vectordb.json`, `docs/benchmarks/browser_ann.json`.

## Server side

| Backend | n | recall@10 | p50 ms | p95 ms | build s | memory MB |
|---|---|---|---|---|---|---|
| numpy brute force | 1e3 | 1.000 | 0.04 | 0.10 | 0.00 | 0.3 |
| faiss Flat | 1e3 | 1.000 | 0.04 | 0.04 | 0.03 | 14.4 |
| faiss HNSW (M 32, efSearch 64) | 1e3 | 1.000 | 0.03 | 0.03 | 0.10 | 0.8 |
| faiss IVF | 1e3 | 0.672 | 0.01 | 0.01 | 0.01 | 4.0 |
| LanceDB IVF-PQ (refine 10) | 1e3 | 1.000 | 3.19 | 4.38 | 1.23 | 159.6 |
| Chroma (HNSW) | 1e3 | 1.000 | 0.58 | 0.96 | 0.55 | 65.7 |
| numpy brute force | 1e4 | 1.000 | 0.15 | 5.97 | 0.00 | 0.0 |
| faiss Flat | 1e4 | 1.000 | 0.31 | 4.00 | 0.00 | 0.0 |
| faiss HNSW | 1e4 | 0.950 | 0.08 | 0.14 | 0.21 | 0.0 |
| faiss IVF | 1e4 | 0.929 | 0.05 | 0.08 | 0.40 | 18.4 |
| LanceDB IVF-PQ | 1e4 | 0.835 | 5.92 | 7.12 | 2.30 | 221.7 |
| Chroma (HNSW) | 1e4 | 0.971 | 0.93 | 1.95 | 0.87 | 61.5 |
| numpy brute force | 1e5 | 1.000 | 3.31 | 17.10 | 0.04 | 153.6 |
| faiss Flat | 1e5 | 1.000 | 5.46 | 19.16 | 0.09 | 153.6 |
| faiss HNSW | 1e5 | 0.836 | 0.13 | 0.24 | 3.07 | 153.6 |
| faiss IVF | 1e5 | 1.000 | 0.41 | 0.52 | 5.20 | 128.2 |
| LanceDB IVF-PQ | 1e5 | 0.670 | 6.04 | 8.42 | 21.42 | 393.1 |
| Chroma (HNSW) | 1e5 | 0.881 | 1.03 | 1.64 | 9.16 | 231.6 |
| numpy brute force | 1e6 | 1.000 | 25.76 | 33.65 | 0.50 | 1536.1 |
| faiss Flat | 1e6 | 1.000 | 49.72 | 84.19 | 1.43 | 1537.0 |
| faiss HNSW | 1e6 | 0.468 | 0.29 | 0.52 | 40.82 | 1797.3 |
| faiss IVF | 1e6 | 1.000 | 4.32 | 4.84 | 51.69 | 1778.4 |
| LanceDB IVF-PQ | 1e6 | 0.765 | 7.06 | 9.23 | 45.42 | 295.0 |

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
