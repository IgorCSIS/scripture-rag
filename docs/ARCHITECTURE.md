# Architecture

![Architecture](../site/assets/architecture.svg)

Modules (`src/scripture_rag/`): `config` (env settings), `sanitize` (upload checks), `ingest` (PDF/TXT to pages), `chunker`, `embed` (Embedder interface, hashing and sentence-transformers), `index` (VectorIndex: brute force, FAISS HNSW, LanceDB), `bm25`, `hybrid` (RRF), `rerank`, `generate` (Generator interface: Ollama, extractive), `validate` (citation regex), `pipeline` (ask flow), `topics`, `similar`, `ia_import`, `eval`, `corpus`.

The browser demo (`site/js/`) reimplements the same pipeline in JavaScript: `core.js` (chunker, BM25, RRF, coverage reranker, k-means, PCA, TF-IDF labels, similar passages), `db.js` (IndexedDB), `app.js` (UI).

## User story map
```mermaid
flowchart TB
  A[As a reader I load a book] --> B[I ask a question in EN/PT/ES]
  B --> C[I get a cited answer or an honest refusal]
  C --> D[I open the quoted passage]
  A --> E[I explore the theme map]
  E --> F[I find similar passages in other traditions]
  C --> G[I search and re-open my history]
```

## Use case
```mermaid
flowchart LR
  R((Reader)) --> UC1[Upload PDF/TXT]
  R --> UC2[Import from Internet Archive]
  R --> UC3[Ask question]
  R --> UC4[View citations]
  R --> UC5[Explore theme map]
  R --> UC6[Find similar passages]
  R --> UC9[Export / import / clear history]
  D((Developer)) --> UC7[Run eval]
  D --> UC8[Run benchmarks]
  UC2 --> IA[(Internet Archive)]
  UC3 --> OL[(Ollama, localhost)]
```

## Architecture
```mermaid
flowchart LR
  U[User] --> UI[Gradio local / Streamlit demo / browser demo]
  UI --> ING[Ingest: sanitize, PDF/TXT parse, page map]
  IA[Internet Archive API] --> ING
  ING --> CH[Chunker: page-aware, overlap]
  CH --> EMB[Embedder: hashing or multilingual-e5-small]
  CH --> BM[BM25 index]
  EMB --> VS[(VectorIndex: brute force, FAISS HNSW, LanceDB)]
  Q[Question] --> HY[Hybrid: BM25 + vector, RRF]
  VS --> HY
  BM --> HY
  HY --> RR[Reranker: lexical or cross-encoder]
  RR --> GATE{Score >= threshold?}
  GATE -- no --> REF[Refusal]
  GATE -- yes --> GEN[Generator: Ollama or extractive]
  GEN --> VER[CitationValidator]
  VER --> UI
  VS --> UNS[TopicModeler: k-means or BERTopic]
  UNS --> MAP[Theme map + similar passages]
  MAP --> UI
```

## Sequence (ask, with refusal branch)
```mermaid
sequenceDiagram
  participant U as User
  participant UI
  participant R as HybridRetriever
  participant RR as Reranker
  participant G as Generator
  participant V as CitationValidator
  U->>UI: question
  UI->>R: retrieve(q, k=50)
  R-->>UI: fused candidates
  UI->>RR: rerank(q, candidates, top 5)
  RR-->>UI: top 5 + scores
  alt top score < REFUSAL_THRESHOLD
    UI-->>U: refusal (no LLM call)
  else
    UI->>G: answer(q, top 5 fenced blocks)
    G-->>UI: draft with citations
    UI->>V: validate(draft, context ids)
    alt invalid
      V-->>UI: errors
      UI-->>U: refusal
    else valid
      UI-->>U: answer + expandable passages
    end
  end
```

## Data flow
```mermaid
flowchart LR
  F[File / IA item] -->|bytes| S[Sanitizer] -->|text by page| C[Chunker] -->|chunks| E[Embedder] -->|vectors| DB[(Vector index)]
  C -->|tokens| B[(BM25)]
  DB --> T[TopicModeler] -->|topic ids, keywords| MAP[Theme map]
  Q[Query] --> H[Hybrid] --> K[Rerank] --> G[Generate] --> V[Validate] --> O[Answer]
  O --> HIST[(Browser: IndexedDB history)]
```

## ERD
```mermaid
erDiagram
  BOOK ||--o{ PAGE : has
  PAGE ||--o{ CHUNK : contains
  CHUNK ||--|| EMBEDDING : has
  TOPIC ||--o{ CHUNK : groups
  QUERY ||--o{ CITATION : produces
  CHUNK ||--o{ CITATION : cited_by
  BOOK { string book_id PK
         string title
         string source
         string license_evidence }
  PAGE { string book_id FK
         int page_no }
  CHUNK { string chunk_id PK
          string book_id FK
          int page_start
          int page_end
          int char_start
          int char_end
          string text }
  EMBEDDING { string chunk_id FK
              string model
              float_array vector }
  TOPIC { int topic_id PK
          string keywords }
  QUERY { int query_id PK
          string question
          bool refused
          float top_score
          int ts }
  CITATION { int query_id FK
             string chunk_id FK }
```
In the browser, IndexedDB stores `books` (BOOK), `chunks` (CHUNK with its EMBEDDING inline), `history` (QUERY with CITATION list inline) and `meta`. In Python, data lives in memory (`RagPipeline.documents`, `chunks`, `BruteForceIndex`); `LanceDBIndex` is the on-disk option. Topics are computed on demand.

## State (query lifecycle)
```mermaid
stateDiagram-v2
  [*] --> Received
  Received --> Retrieving
  Retrieving --> Reranking
  Reranking --> Refused: score < threshold
  Reranking --> Generating
  Generating --> Refused: generator refused
  Generating --> Validating
  Validating --> Answered: citations valid
  Validating --> Refused: invalid
  Answered --> [*]
  Refused --> [*]
```

## Class
```mermaid
classDiagram
  class Document { +book_id +title +pages +source +license_evidence }
  class Chunk { +chunk_id +book_id +text +page_start +page_end +char_start +char_end }
  class Chunker { +split(doc) list~Chunk~ }
  class Embedder { <<interface>> +encode(texts, is_query) ndarray }
  class HashingEmbedder
  class SentenceTransformerEmbedder
  class VectorIndex { <<interface>> +add(ids, vecs) +search(v, k) }
  class BruteForceIndex
  class FaissHNSWIndex
  class LanceDBIndex
  class BM25Index { +add(ids, texts) +search(q, k) }
  class HybridRetriever { +retrieve(q, k) +vector_only(q, k) +bm25_only(q, k) }
  class Reranker { <<interface>> +score(q, passages) +rerank(q, chunks, n) }
  class LexicalOverlapReranker
  class CrossEncoderReranker
  class Generator { <<interface>> +answer(q, ctx) }
  class OllamaGenerator
  class ExtractiveGenerator
  class CitationValidator { +validate(answer, ids) ValidationResult }
  class TopicModeler { +fit(vecs, texts) TopicResult +project(vecs) }
  class RagPipeline { +add_document(doc) +ask(q) Answer }
  Embedder <|.. HashingEmbedder
  Embedder <|.. SentenceTransformerEmbedder
  VectorIndex <|.. BruteForceIndex
  VectorIndex <|.. FaissHNSWIndex
  VectorIndex <|.. LanceDBIndex
  Reranker <|.. LexicalOverlapReranker
  Reranker <|.. CrossEncoderReranker
  Generator <|.. OllamaGenerator
  Generator <|.. ExtractiveGenerator
  HybridRetriever --> VectorIndex
  HybridRetriever --> BM25Index
  HybridRetriever --> Embedder
  Chunker --> Chunk
  RagPipeline --> HybridRetriever
  RagPipeline --> Reranker
  RagPipeline --> Generator
  RagPipeline --> CitationValidator
```

## GUI mockup
```mermaid
flowchart TB
  subgraph App[Scripture RAG]
    direction TB
    subgraph Left[Library panel]
      L1[Upload PDF/TXT]
      L2[IA identifier + Check license + Import]
      L3[Book list with license evidence]
      L4[Export / Import JSON / Clear all]
    end
    subgraph Main[Tabs]
      T1[Ask: question box]
      T2[Sources: expandable passages + PDF page]
      T3[Theme Map: scatter + topic legend + book shapes]
      T4[Similar Passages: other books]
      T5[History: search + re-open]
    end
  end
```

## Diagrams as SVG
`site/assets/pipeline.svg`, `retrieval.svg`, `clustering.svg`, `architecture.svg` are hand-written SVG versions used in the README and the website.
