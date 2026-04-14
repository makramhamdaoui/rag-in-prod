# RAG in Production

A local, private Retrieval-Augmented Generation (RAG) system built with:

- **OpenSearch** — hybrid search (BM25 + KNN vector search)
- **SentenceTransformers** — document and query embeddings
- **Ollama** — local LLM inference (qwen3:8b)
- **FastAPI** — REST API with per-session chat history

