"""
Knat LLM — Project 4: retrieval-augmented generation query pipeline.

Wires an Elasticsearch embedding retriever to a locally served (or hosted) LLM
via an OpenAI-compatible generator, with a lightweight tracing hook so each
query's retrieval + generation steps can be inspected downstream.
"""
import os
import time

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question using only the context below. If the answer isn't in the
context, say you don't know.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""


def build_rag_pipeline() -> Pipeline:
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    index_name = os.getenv("ELASTICSEARCH_INDEX", "knat_rag_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    llm_api_key = os.getenv("LLM_API_KEY", "not-needed")

    document_store = ElasticsearchDocumentStore(hosts=es_url, index=index_name)

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=embedding_model))
    pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=5))
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component(
        "generator",
        OpenAIGenerator(api_base_url=llm_base_url, model=llm_model, api_key=llm_api_key),
    )

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")
    return pipeline


def ask(pipeline: Pipeline, question: str) -> dict:
    """Runs a query and returns the answer plus lightweight trace/timing info.

    In production, replace the manual timing below with an OpenTelemetry span
    (see the OTEL_EXPORTER_OTLP_ENDPOINT setting in .env.example) so retrieval
    and generation latency show up in a real tracing backend (e.g. Jaeger,
    Langfuse, or an OTLP-compatible APM).
    """
    start = time.perf_counter()
    result = pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
        }
    )
    elapsed = time.perf_counter() - start
    answer = result["generator"]["replies"][0]
    return {"answer": answer, "latency_seconds": round(elapsed, 3)}
