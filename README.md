# LLM-Powered Document Search (RAG)

Ask questions about your own documents in plain English and get answers drawn from them, with
citations, instead of keyword matching. Every answer is grounded in the source text, so responses are
specific and traceable rather than generic LLM guesses. Uploading a PDF automatically makes it
searchable, with no manual indexing step.

**Stack:** Python, LangChain, FAISS, ChromaDB, Ollama

## How it works

```mermaid
flowchart LR
    U[PDF / MD / TXT upload<br/>or watched folder] --> L[loader<br/>page-aware]
    L --> S[RecursiveCharacterTextSplitter]
    S --> E[Ollama embeddings<br/>nomic-embed-text]
    E --> V[(FAISS or ChromaDB<br/>persistent)]
    Q[Question] --> R[retrieve top-k + relevance filter]
    V --> R
    R -->|nothing relevant| N["I couldn't find this in your documents"]
    R --> G[ChatOllama<br/>grounded prompt] --> A[Answer with [1] [2] citations]
```

- **Grounded answers:** the model may only use the numbered passages, must cite them as `[n]`, and
  must reply "I couldn't find this in your documents." otherwise. If retrieval finds nothing above
  `MIN_RELEVANCE`, the LLM is not even called.
- **Traceable:** each answer returns the source file, page number, similarity score and snippet, and flags which
  passages were actually cited.
- **Automated ingestion:** `POST /documents` (or dropping a file in a watched folder) loads, chunks,
  embeds and indexes it in one step. Ingestion is idempotent via a content-hash manifest: unchanged
  files are skipped, edited files replace their old chunks, deleted files are purged.
- **Two interchangeable vector stores:** FAISS (default) and ChromaDB via `VECTOR_STORE`.
  Embeddings are L2-normalised so relevance is cosine similarity in both.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 1. try it with no model (hash embeddings + extractive answers)
export RAG_BACKEND=offline
python -m docsearch.cli ingest sample_docs
python -m docsearch.cli ask "How many days of sick leave do employees get?"

# 2. run it for real with local models
ollama pull nomic-embed-text && ollama pull llama3.1:8b
unset RAG_BACKEND
python -m docsearch.cli ingest path/to/your.pdf
python -m docsearch.cli ask "What is the notice period?"
python -m docsearch.cli watch data/inbox      # auto-index anything dropped here
```

### Web UI

```bash
python -m docsearch.cli serve --port 8000     # open http://localhost:8000
```

Drag files into the sidebar to index them, ask a question, and click a `[1]` citation chip to jump to the
exact source passage, with its page number and similarity score.

### API

```bash
python -m docsearch.cli serve --port 8000
curl -F "file=@handbook.pdf" localhost:8000/documents
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How much are meals reimbursed per day?"}'
```

Interactive docs are at `http://localhost:8000/docs`. Endpoints: `POST /documents`,
`GET /documents`, `DELETE /documents/{name}`, `POST /ask`, `GET /health`.

## Configuration

See `.env.example`: `RAG_BACKEND`, `VECTOR_STORE`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_RELEVANCE`.

## Notes and limits

- Scanned PDFs with no text layer are rejected with a clear error (OCR is not included).
- The FAISS index is stored with pickle metadata. Only load indexes that this app created.
- If `faiss-cpu` has no wheel for your platform, set `VECTOR_STORE=chroma` or install FAISS from conda.
- Run the tests with `pytest` (both vector stores are covered, plus PDF upload through the API).

## License

MIT
