# Project 4 — RAG Pipeline: PDFs → Embeddings → Elasticsearch → Haystack → Chat UI

**Org:** Knat LLM · **Maintained by:** Knatware Technology

A complete retrieval-augmented generation stack: PDF ingestion, embedding generation, storage and
semantic search in **Elasticsearch**, orchestration with **Haystack**, and a simple **chat UI**,
with a tracing hook for observability. This is where the model-serving work from Projects 2–3
becomes an actual application people can use.

## Architecture

```
 PDFs ──▶ PyPDFToDocument ──▶ Cleaner ──▶ Splitter ──▶ Embedder ──▶ Elasticsearch (writer)
(data/raw)      (Haystack indexing pipeline — ingestion/ingest.py)

                                                     ┌─────────────────────────┐
User question ──▶ Text Embedder ──▶ ES Retriever ──▶│  Prompt Builder          │
                                                     │  (context + question)    │
                                                     └───────────┬─────────────┘
                                                                 ▼
                                                 OpenAI-compatible Generator
                                                (points at the vLLM server from
                                                          Project 2)
                                                                 │
                                                                 ▼
                                                        Streamlit chat UI
```

## Components

| Layer            | Tool                                  | File                         |
|-------------------|----------------------------------------|------------------------------|
| PDF ingestion      | Haystack `PyPDFToDocument`, splitter    | `ingestion/ingest.py`         |
| Embeddings         | `sentence-transformers` (MiniLM)        | shared config via `.env`      |
| Vector store       | Elasticsearch (dense vector search)     | `docker-compose.yml`          |
| Orchestration      | Haystack query pipeline                 | `pipeline/rag_pipeline.py`     |
| Generation         | OpenAI-compatible client → vLLM server  | `pipeline/rag_pipeline.py`     |
| UI                 | Streamlit chat interface                 | `ui/app.py`                    |
| Tracing            | Manual latency capture, OTLP-ready       | `pipeline/rag_pipeline.py`     |

## Getting started

```bash
git clone <this-repo-url>
cd project-04-rag-pipeline
cp .env.example .env
```

Update `.env` so `LLM_BASE_URL` points at a running OpenAI-compatible server — by default this
assumes the vLLM server from **Project 2** is running locally and reachable via
`host.docker.internal`.

```bash
docker compose up --build -d       # brings up Elasticsearch + the UI container
```

Add PDFs to `data/raw/`, then run ingestion (inside the `rag-ui` container or a local venv with
the same requirements installed):

```bash
docker compose exec rag-ui python ingestion/ingest.py --input-dir data/raw
```

Open the chat UI at `http://localhost:8501` and ask questions grounded in the ingested documents.

## How retrieval quality is kept honest

The prompt template in `pipeline/rag_pipeline.py` explicitly instructs the model to say it
doesn't know rather than guess when the retrieved context doesn't contain the answer — a small
but important detail for enterprise RAG systems, where a confident wrong answer is worse than an
honest "I don't know."

## Tracing

`ask()` currently records wall-clock latency per query as a lightweight starting point. The
`OTEL_EXPORTER_OTLP_ENDPOINT` variable in `.env.example` is there for the natural next step:
wrapping the retrieval and generation steps in **OpenTelemetry spans** and exporting them to a
real backend (Jaeger, Langfuse, or any OTLP-compatible APM) so retrieval time and generation time
are visible as separate spans per request — critical for diagnosing whether a slow answer is a
search problem or a model problem.

## What this project deliberately practices

- Building a real Haystack indexing pipeline (convert → clean → split → embed → write)
- Using Elasticsearch as a dense vector store, not just a full-text search engine
- Separating retrieval from generation so either can be swapped independently
- Prompting for groundedness/refusal rather than hallucination
- Thinking about tracing and observability from the start, not bolted on later

## Natural next steps

- Add re-ranking (cross-encoder) between retrieval and generation for higher precision
- Add per-user conversation history in PostgreSQL (tie back to Project 1's patterns)
- Swap the manual latency capture for full OpenTelemetry spans exported to Jaeger/Langfuse
- Add a relevance-evaluation harness (e.g. RAGAS) to score answer groundedness over time

## Enquiries & implementation support

For enquiries, custom implementation, or extending this into a multi-tenant RAG platform, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
