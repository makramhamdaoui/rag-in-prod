# RAG in Production

A fully local, private RAG system. Documents are indexed into OpenSearch with hybrid search (BM25 + KNN). 

Queries go through an agentic pipeline: guardrail → retrieve → grade → rewrite → generate,running on a local LLM via Ollama. 

Responses are cached in Redis using exact and semantic matching.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM | Ollama (`qwen3:8b` by default) |
| Embeddings | SentenceTransformers (`all-mpnet-base-v2`) |
| Search | OpenSearch — hybrid BM25 + KNN |
| PDF parsing | Docling |
| Database | PostgreSQL — document metadata |
| Cache | Redis — exact + semantic response cache |
| Tracing | Langfuse (optional) |
| Agent framework | LangGraph |

## Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.com) running locally on port `11434`
- Python 3.12+ with a virtual environment

Pull the model before starting:
```bash
ollama pull qwen3:8b
```

## Quick start

```bash
# 1. clone and enter the project
git clone <repo-url> && cd rag-in-prod

# 2. copy secrets
cp .env.example .env          # edit POSTGRES_PASSWORD etc. if needed

# 3. start all infrastructure
make start

# 4. install Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 5. start the API
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Configuration

Non-secret settings live in `config.yaml`, edit it directly.

```yaml
ollama:
  model_name: "qwen3:8b"   # swap to any model you have pulled

embedding:
  model_path: "sentence-transformers/all-mpnet-base-v2"
  dimension: 768
  asymmetric: false          # set true for passage/query asymmetric models

chunking:
  chunk_size: 300            # words per chunk
  overlap: 100

agents:
  guardrail_threshold: 75    # 0-100, queries below this score are rejected
  max_retrieval_attempts: 2  # retries with rewritten query before giving up

cache:
  ttl_seconds: 86400         # 24 h
  similarity_threshold: 0.85 # cosine threshold for semantic cache hits
```

Secrets (passwords, API keys) go in `.env` only — they are never read from `config.yaml`:

```
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
LANGFUSE_PUBLIC_KEY=        # leave empty to disable tracing
LANGFUSE_SECRET_KEY=
```

## Ingesting documents

Drop a PDF into `uploaded_files/` then call the ingest endpoint:

```bash
curl -X POST "http://localhost:8000/documents/ingest?file_path=uploaded_files/my-doc.pdf"
```

What happens internally:
1. Docling parses the PDF into sections with page numbers
2. Sections are chunked into 300-word overlapping windows
3. Each chunk is embedded with SentenceTransformers
4. Chunks + embeddings are bulk-indexed into OpenSearch
5. Document record is written to PostgreSQL with status `indexed`

List indexed documents:
```bash
curl http://localhost:8000/documents
```

Delete a document (removes chunks from OpenSearch and the record from Postgres):
```bash
curl -X DELETE http://localhost:8000/documents/<document_id>
```

## API reference

### Chat

```
POST /chat
```
Streaming RAG response. Returns `text/plain` chunked stream.

```json
{
  "query": "What is the Transformer architecture?",
  "session_id": "optional-uuid",
  "use_hybrid_search": true,
  "num_results": 3,
  "temperature": 0.3
}
```

```
POST /chat/cached
```
Same as `/chat` but checks Redis first (exact key, then cosine similarity). Returns JSON with `cached`, `match_type`, and `similarity` fields.

### Agentic RAG

```
POST /agentic
```
Full LangGraph pipeline: guardrail → retrieve → grade → rewrite → generate.

```json
{ "query": "Explain attention mechanisms", "session_id": "optional-uuid" }
```

Response includes `reasoning_steps`, `sources`, `rewritten_query`, `guardrail_score`, and `execution_time`.

### Sessions

```
POST   /session                      → create session, returns session_id
GET    /session/{session_id}/history → get message history
DELETE /session/{session_id}         → clear session
```

### Documents

```
GET    /documents                          → list all indexed documents
POST   /documents/ingest?file_path=<path>  → ingest a PDF
DELETE /documents/{document_id}            → delete document + chunks
```

### Health

```
GET /health   → {"status": "ok", "model": "qwen3:8b"}
```

## Agentic pipeline

```
query
  │
  ▼
guardrail          score 0–100 against indexed doc names
  │                score < 75 → out_of_scope response
  ▼
retrieve           hybrid search (BM25 + KNN), top-5 chunks
  │
  ▼
grade_documents    LLM grades chunk relevance (yes/no)
  │                not relevant + attempts < 2 → rewrite_query
  ▼
generate_answer    LLM answers from graded context
```

The guardrail queries PostgreSQL for currently indexed document names so it stays accurate as documents are added or removed.

## Semantic cache

Every `/chat/cached` call is checked in two stages:

1. **Exact match** — SHA-256 hash of `(query, num_results, use_hybrid_search)`. O(1).
2. **Semantic match** — cosine similarity between query embedding and all cached embeddings. Hit if similarity ≥ `0.85` (configurable).

Cache TTL defaults to 24 h. Both keys are written on every cache miss.

## Observability (Langfuse)

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`. Langfuse runs locally via compose on `http://localhost:3000`.

Default credentials: `admin@example.com` / `admin123`.

Each RAG query creates a trace with:
- `hybrid_search` span — retrieved chunks + scores
- `ollama_generation` span — model, latency, response
- Cache hit metadata (type + similarity score)

Leave the keys empty to run without tracing — it degrades gracefully.

## Project structure

```
├── api/                        FastAPI app factory and routers
│   ├── routes/
│   │   ├── chat.py             /chat, /chat/cached
│   │   ├── agentic.py          /agentic
│   │   ├── documents.py        /documents
│   │   ├── sessions.py         /session
│   │   └── health.py           /health
│   └── schemas.py
├── src/
│   ├── config.py               Settings (YAML + .env)
│   ├── logging.py              JSON + human formatters, correlation IDs
│   ├── constants.py            Flat constants derived from Settings
│   ├── database/
│   │   ├── connection.py       SQLAlchemy engine + session
│   │   └── models/document.py  Document ORM model
│   └── services/
│       ├── llm/
│       │   ├── client.py       Ollama wrapper, streaming, RAG pipeline
│       │   └── prompts.py      Prompt assembly
│       ├── search/
│       │   ├── client.py       OpenSearch client
│       │   ├── hybrid_search.py BM25 + KNN hybrid search
│       │   └── ingestion.py    Index management, bulk indexing
│       ├── embeddings/
│       │   └── client.py       SentenceTransformer wrapper
│       ├── chunking/
│       │   └── chunker.py      Section chunker + text cleaner
│       ├── pdf_parser/
│       │   ├── docling_parser.py Docling-based PDF → sections
│       │   └── ocr.py          PyPDF2 + Tesseract fallback
│       ├── agents/             LangGraph agentic pipeline
│       │   ├── graph.py        Graph definition + run_agentic_rag()
│       │   ├── state.py        AgentState TypedDict
│       │   ├── prompts.py      Agent prompts
│       │   └── nodes/          guardrail, retrieve, grade, rewrite, generate
│       ├── cache/client.py     Redis exact + semantic cache
│       └── tracing/client.py   Langfuse RAGTracer
├── config/
│   └── opensearch_index.json   Index mapping + HNSW settings
├── config.yaml                 Non-secret runtime settings
├── .env                        Secrets (git-ignored)
├── main.py                     Uvicorn entry point
├── Makefile                    Shortcuts for common operations
└── tests/
```

## Make commands

```bash
make start      # docker compose up -d
make stop       # docker compose down
make restart    # docker compose restart
make status     # docker compose ps
make logs       # tail -f compose logs
make health     # curl all services and report status
make clean      # down -v (destroys volumes)
```

## Running tests

```bash
source .venv/bin/activate
python -m pytest -v

# unit tests only (no services needed)
python -m pytest tests/test_cache.py -v -k "not store_and_retrieve"
```

Integration tests skip automatically when Postgres/Redis/OpenSearch are not reachable.
