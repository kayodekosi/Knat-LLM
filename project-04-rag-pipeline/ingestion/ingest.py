"""
Knat LLM — Project 4: PDF ingestion into Elasticsearch via a Haystack indexing pipeline.

Usage:
    python ingestion/ingest.py --input-dir data/raw
"""
import argparse
import os

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from dotenv import load_dotenv

load_dotenv()


def build_indexing_pipeline(document_store: ElasticsearchDocumentStore, embedding_model: str) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5, split_overlap=1))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedding_model))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    args = parser.parse_args()

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    index_name = os.getenv("ELASTICSEARCH_INDEX", "knat_rag_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    document_store = ElasticsearchDocumentStore(hosts=es_url, index=index_name)
    pdf_paths = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        print(f"No PDFs found in {args.input_dir}. Add files and re-run.")
        return

    pipeline = build_indexing_pipeline(document_store, embedding_model)
    pipeline.run({"converter": {"sources": pdf_paths}})

    print(f"Ingested {len(pdf_paths)} PDF(s) into index '{index_name}' at {es_url}")


if __name__ == "__main__":
    main()
