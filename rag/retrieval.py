from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import settings


def initialize_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        num_gpu=settings.OLLAMA_NUM_GPU,
    )
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
    )


def retrieve_documents(query: str, top_k: int | None = None) -> list[Document]:
    vector_store = initialize_vector_store()
    if not vector_store.get(include=["metadatas"]).get("ids"):
        return []
    limit = top_k or settings.RETRIEVAL_TOP_K
    return vector_store.similarity_search(query, k=limit)


def build_source_references(documents: list[Document]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[tuple[str, int | str, str]] = set()

    for document in documents:
        metadata = document.metadata or {}
        filename = metadata.get("source_filename", "Unknown source")
        page_number = metadata.get("page_number")
        snippet = (document.page_content or "").strip()
        excerpt = snippet[:180].replace("\n", " ")
        key = (filename, page_number or "unknown", snippet[:120])
        if key in seen:
            continue
        seen.add(key)
        reference = {
            "filename": filename,
            "page_number": page_number,
            "excerpt": excerpt,
        }
        references.append(reference)

    return references


def list_indexed_documents() -> list[dict[str, Any]]:
    vector_store = initialize_vector_store()
    try:
        results = vector_store.get(include=["metadatas", "documents"])
    except Exception:
        return []

    metadata_items = results.get("metadatas", [])
    if not metadata_items:
        return []

    unique_documents: dict[str, dict[str, Any]] = {}
    for metadata in metadata_items:
        document_id = metadata.get("document_id") or metadata.get("source_filename")
        if not document_id:
            continue
        current = unique_documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": metadata.get("source_filename", "Unknown source"),
                "page_numbers": set(),
                "chunk_count": 0,
            },
        )
        current["page_numbers"].add(metadata.get("page_number", "Unknown"))
        current["chunk_count"] += 1

    entries: list[dict[str, Any]] = []
    for value in unique_documents.values():
        page_numbers = sorted(value["page_numbers"], key=lambda item: str(item))
        entries.append(
            {
                "document_id": value["document_id"],
                "filename": value["filename"],
                "page_numbers": page_numbers,
                "chunk_count": value["chunk_count"],
            }
        )
    return entries
