# Semantic RAG & Vector Search Assessment

A local Retrieval-Augmented Generation (RAG) pipeline that benchmarks two retrieval strategies:

- **Strategy A (Raw Vector Search):** Direct embedding similarity search
- **Strategy B (AI-Enhanced Retrieval):** Query expansion via a (mocked) generative model before search

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           RAGPipeline                │
                    │  ingest() / search(strategy="A"|"B") │
                    └─────────┬───────────────┬───────────┘
                              │               │
                   Strategy A │    Strategy B  │
                              │               │
                    ┌─────────▼──┐  ┌─────────▼──────────┐
                    │ RawVector  │  │ AIEnhancedSearch    │
                    │ Search     │  │  ┌────────────────┐ │
                    │            │  │  │ QueryExpander  │ │
                    │            │  │  └───────┬────────┘ │
                    └─────┬──────┘  └──────────┬─────────┘
                          │                    │
                    ┌─────▼────────────────────▼─────────┐
                    │     LocalEmbeddingModel             │
                    │     (sentence-transformers)         │
                    └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │       FAISSVectorStore              │
                    │  (IndexFlatIP, cosine similarity)   │
                    └────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest -v

# Run benchmark
python scripts/run_benchmark.py
```

## Project Structure

```
├── data/
│   └── corpus.py              # 10 technical paragraphs about "ArcScale" distributed system
├── src/
│   ├── embedding/
│   │   ├── base.py            # BaseEmbeddingModel interface (mirrors Vertex AI API)
│   │   ├── local_embedder.py  # sentence-transformers implementation
│   │   └── vertex_adapter.py  # Production Vertex AI adapter (for docs + mock testing)
│   ├── storage/
│   │   ├── base.py            # BaseVectorStore interface
│   │   └── faiss_store.py     # FAISS IndexFlatIP with cosine similarity
│   ├── retrieval/
│   │   ├── base.py            # RetrievalStrategy interface
│   │   ├── raw_vector_search.py     # Strategy A
│   │   ├── ai_enhanced_search.py    # Strategy B
│   │   ├── query_expander.py        # Local (synonym map) + Vertex (GenerativeModel)
│   │   └── pipeline.py             # Orchestrator
│   └── benchmark/
│       ├── queries.py         # 5 benchmark queries with ground-truth relevance
│       ├── metrics.py         # Precision@k, Recall@k, MRR
│       └── runner.py          # BenchmarkRunner and report generation
├── tests/
│   ├── test_embedding.py      # Embedding model tests
│   ├── test_storage.py        # FAISS store tests
│   ├── test_retrieval.py      # Strategy A/B + pipeline tests
│   ├── test_query_expander.py # Query expansion tests
│   ├── test_vertex_mocks.py   # Vertex AI SDK mocking demonstration
│   └── test_benchmark.py      # Benchmark runner + metrics tests
├── scripts/
│   └── run_benchmark.py       # Entry point
└── retrieval_benchmark.md     # Full benchmark output and analysis
```

## Similarity Metric Choice

**Cosine similarity** (via inner product on L2-normalized vectors):

1. `all-MiniLM-L6-v2` is trained with a cosine similarity objective — using the matching metric produces the most meaningful results
2. Cosine measures angular proximity (semantic direction), not magnitude — correct for text embeddings
3. Bounded output [-1, 1] (effectively [0, 1] for text embeddings) enables meaningful confidence thresholds
4. For unit vectors: `cosine_sim(a,b) = dot(a,b)` and `L2_dist = sqrt(2 - 2*cosine_sim)` — the two metrics are monotonically related, but cosine gives more interpretable scores

**Implementation:** FAISS `IndexFlatIP` computes inner product. Vectors are L2-normalized both at embedding time (`normalize_embeddings=True`) and at index time (`faiss.normalize_L2()`), making inner product equivalent to cosine similarity.

## Vertex AI Migration Path

> **SDK Note:** `vertexai.language_models` and `vertexai.generative_models` were deprecated on June 24, 2025 (removed June 2026). This codebase includes legacy adapters for mock testing per the assessment spec, **plus** production-ready adapters using the recommended `google-genai` SDK.

| Component          | Local                    | Production (google-genai)                       | Legacy (vertexai, deprecated)         |
|--------------------|--------------------------|--------------------------------------------------|---------------------------------------|
| Embeddings         | `LocalEmbeddingModel`    | `GenAIEmbeddingAdapter` (gemini-embedding-001)   | `VertexEmbeddingAdapter` (text-embedding-005) |
| Query Expansion    | `LocalQueryExpander`     | `GenAIQueryExpander` (Gemini 2.5 Flash / 3 Flash)          | `VertexQueryExpander`                 |
| Vector Store       | `FAISSVectorStore`       | Vertex AI Matching Engine (`google.cloud.aiplatform` — not deprecated) | — |
| Serving            | Local script             | Cloud Run + FastAPI                              | —                                     |

Each component implements an abstract interface (`BaseEmbeddingModel`, `BaseQueryExpander`, `BaseVectorStore`), so the migration requires swapping implementations with zero changes to the retrieval pipeline logic.

See [retrieval_benchmark.md](retrieval_benchmark.md) for detailed migration steps and architecture diagrams.

## Design Decisions

- **IndexFlatIP over IndexFlatL2:** Inner product on normalized vectors = cosine similarity. Returns similarity scores (higher is better) rather than distances, matching Vertex AI Matching Engine defaults.
- **Double normalization:** Vectors are normalized at both embedding and index time as a defensive pattern — ensures cosine semantics even if a different embedder is plugged in.
- **Lazy SDK imports:** All Vertex AI and Gen AI adapters import their SDK inside `__init__()`, not at module level. This allows the modules to be imported and tested via mock without `google-cloud-aiplatform` or `google-genai` installed.
- **Real synonym injection (not passthrough):** `LocalQueryExpander` implements genuine query expansion with domain-specific synonyms, producing a meaningful benchmark comparison rather than a trivial identity transform.

## Testing

```bash
# All tests
pytest -v

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_vertex_mocks.py -v
```

33 tests covering:
- Embedding model correctness and normalization
- FAISS store CRUD and edge cases
- Both retrieval strategies
- Query expansion logic and deduplication
- Vertex AI SDK mocking (TextEmbeddingModel + GenerativeModel)
- Benchmark runner, metrics, and output formatting
