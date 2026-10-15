# Architecture

![Architecture: browser app and Python package, kept in parity](diagrams/architecture.svg)

Answers are extractive: every answer is a set of quoted verses with citations, or a refusal. There is no LLM or text generation step anywhere in the pipeline.

Modules (`src/scripture_rag/`): `config` (env settings), `sanitize` (upload checks), `ingest` (PDF/TXT to pages), `verses` (verse parsing and verse chunks), `chunker` (fallback token windows), `normalize` (archaic normalization and query expansion), `text_utils` (tokenizer, `clean_query`), `translate` (language detection, PT/ES to EN MarianMT), `embed`, `index` (brute force, FAISS HNSW, LanceDB), `bm25`, `hybrid` (weighted RRF), `rerank` (cross-encoder, lexical fallback), `gate` (per-language score gate, lexical support), `generate` (extractive answer builder), `validate` (citation check), `pipeline` (ask flow), `topics`, `similar`, `library` (manifest), `ia_search`, `ia_import`, `discover`, `pdfgen`, `eval`, `corpus`.

The browser app (`site/js/`) runs the same pipeline with the same defaults: `rag.js` (pipeline), `models.js` (transformers.js models), `ann.js` (HNSW above 20k passages), `core.js` (BM25, k-means, PCA, TF-IDF labels), `library.js` (manifest, roles, compare), `ia.js` and `archive-ui.js` (Internet Archive panel), `db.js` (IndexedDB), `app.js` and `ui.js` (UI). Browser and Python make identical answer/refuse decisions and citations on the eval set ([docs/eval/parity.json](eval/parity.json)).

## User story map
```mermaid
flowchart TB
  A[As a reader I load a book] --> B[I ask a question in EN, PT or ES]
  B --> C[I get cited verse quotes or an honest refusal]
  C --> D[I open the quoted passage]
  A --> E[I explore the theme map]
  E --> F[I find similar passages in other traditions]
  C --> G[I compare answers across traditions]
  C --> H[I search and re-open my history]
```

## Use case
```mermaid
flowchart LR
  R((Reader)) --> UC1[Upload PDF/TXT]
  R --> UC2[Search and import from Internet Archive]
  R --> UC3[Ask question]
  R --> UC4[View citations]
  R --> UC5[Explore theme map]
  R --> UC10[Compare by tradition]
  R --> UC9[Export / import / clear history]
  D((Developer)) --> UC7[Run eval]
  D --> UC8[Run benchmarks]
  UC2 --> IA[(Internet Archive)]
  UC3 --> HF[(Hugging Face models, downloaded once)]
```

## Architecture
```mermaid
flowchart TB
  subgraph Ingest
    direction LR
    F[PDF / TXT or IA item] --> S[Sanitize + parse pages] --> N[Normalize archaic] --> CH[Verse chunks, fallback windows]
    CH --> BM[(BM25)]
    CH --> VS[(Vectors: brute force, HNSW above 20k)]
  end
  subgraph Query
    direction LR
    Q[Question] --> CQ[clean_query] --> LD[Detect language] --> TR[Translate PT/ES to EN] --> EX[Normalize + expand]
  end
  subgraph Rank
    direction LR
    H[BM25 + e5 dense] --> RRF[Weighted RRF] --> RR[Rerank top 50] --> G1{score >= t lang?} -- yes --> G2{support >= 0.3?}
  end
  EX --> H
  BM --> H
  VS --> H
  G2 -- yes --> ANS[Extractive answer: cited verse quotes]
  G1 -- no --> REF[Refusal]
  G2 -- no --> REF
  VS --> TM[Theme map: k-means, PCA, outliers]
```

## Sequence (ask, with refusal branch)
```mermaid
sequenceDiagram
  participant U as User
  participant P as Pipeline
  participant T as Translator
  participant R as HybridRetriever
  participant RR as Reranker
  participant G as Gates
  U->>P: question
  P->>P: clean_query, detect language
  P->>T: PT/ES question
  T-->>P: English query
  P->>R: expanded query + original query
  R-->>P: top 50 fused by weighted RRF
  P->>RR: rerank 50 candidates
  RR-->>P: top 5 + scores
  P->>G: score vs t[lang], lexical support
  alt below either gate
    P-->>U: I could not find this in the loaded texts.
  else both gates pass
    P-->>U: top 3 verse quotes, each cited [book p.N #id]
  end
```

## Data flow
```mermaid
flowchart TB
  F[File / IA item] -->|bytes| S[Sanitizer] -->|text by page| N[Normalize] -->|text| C[Verse chunker] -->|chunks| E[Embedder] -->|vectors| DB[(Vector index)]
  C -->|tokens| B[(BM25)]
  DB --> T[Theme map] -->|clusters, keywords| MAP[UI]
  Q[Query] --> CL[Clean + translate] --> X[Expand] --> H[Hybrid RRF] --> K[Rerank 50] --> GT[Lang gate + support gate] --> O[Cited quotes or refusal]
  B --> H
  DB --> H
  O --> HIST[(IndexedDB history)]
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
  [*] --> Preparing: ask
  Preparing --> Retrieving: clean, detect, translate
  Retrieving --> Reranking: top 50
  Reranking --> Gating
  Gating --> Refused: score < t[lang]
  Gating --> SupportCheck: score ok
  SupportCheck --> Refused: support < 0.3
  SupportCheck --> Answered: quote + validate
  Answered --> Saved
  Refused --> Saved
  Saved --> [*]
```

## Class
```mermaid
classDiagram
  direction LR
  class Document {
    +book_id
    +title
    +pages
    +source
    +license_evidence
  }
  class Chunk {
    +chunk_id
    +book_id
    +text
    +page_start
    +page_end
    +ref
  }
  class Embedder {
    <<interface>>
    +encode(texts, is_query) ndarray
  }
  class VectorIndex {
    <<interface>>
    +add(ids, vecs)
    +search(v, k)
  }
  class BM25Index {
    +add(ids, texts)
    +search(q, k)
  }
  class HybridRetriever {
    +retrieve(q, k)
  }
  class QueryTranslator {
    +to_english(text)
  }
  class ArchaicNormalizer {
    +text(text)
    +word(w)
  }
  class Reranker {
    <<interface>>
    +rerank(q, chunks, n)
  }
  class CrossEncoderReranker
  class LexicalOverlapReranker
  class GatePolicy {
    +thresholds
    +min_support
    +threshold(lang)
    +allow(lang, score, sup)
  }
  class ExtractiveAnswer {
    +answer(q, ctx)
  }
  class CitationValidator {
    +validate(answer, ids)
  }
  class RagPipeline {
    +add_document(doc)
    +ask(q) Answer
  }
  Reranker <|.. CrossEncoderReranker
  Reranker <|.. LexicalOverlapReranker
  HybridRetriever --> VectorIndex
  HybridRetriever --> BM25Index
  HybridRetriever --> Embedder
  RagPipeline --> QueryTranslator
  RagPipeline --> ArchaicNormalizer
  RagPipeline --> HybridRetriever
  RagPipeline --> Reranker
  RagPipeline --> GatePolicy
  RagPipeline --> ExtractiveAnswer
  RagPipeline --> CitationValidator
  RagPipeline --> Chunk
  Document --> Chunk
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
`docs/diagrams/architecture.svg`, `data_flow.svg`, `state.svg` and `site/assets/pipeline.svg`, `retrieval.svg`, `clustering.svg`, `architecture.svg` share one style (system font stack, rounded cards on a light panel that stays readable on light and dark backgrounds). They are used in the README and on the website.
