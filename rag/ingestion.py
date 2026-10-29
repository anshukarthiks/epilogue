from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.retrieval import initialize_vector_store


def ensure_storage_directories() -> None:
    Path(settings.UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    safe_name = os.path.basename(filename).strip()
    if not safe_name:
        raise ValueError("No filename was provided for the uploaded file.")
    if any(segment in {"..", "/", "\\"} for segment in safe_name.split(".")):
        raise ValueError("Uploaded file names must be safe and cannot include path traversal patterns.")
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return safe_name


def compute_document_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def document_already_indexed(document_hash: str) -> bool:
    vector_store = initialize_vector_store()
    try:
        existing = vector_store.get(where={"document_id": document_hash})
    except Exception:
        return False
    return bool(existing and existing.get("ids"))


def extract_pdf_pages(file_path: str) -> list[dict[str, Any]]:
    document = fitz.open(file_path)
    pages: list[dict[str, Any]] = []
    try:
        for page_number in range(document.page_count):
            text = document[page_number].get_text("text")
            if text and text.strip():
                pages.append({"page_number": page_number + 1, "text": text.strip()})
    finally:
        document.close()

    if not pages:
        raise ValueError("The uploaded PDF contains no extractable text.")
    return pages


def split_pages_into_chunks(pages: list[dict[str, Any]], document_id: str, filename: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = []

    for page in pages:
        page_number = page["page_number"]
        texts = splitter.split_text(page["text"])
        for chunk_index, chunk_text in enumerate(texts):
            metadata = {
                "document_id": document_id,
                "source_filename": filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_id": f"{document_id}-{page_number}-{chunk_index}",
                "document_hash": document_id,
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not chunks:
        raise ValueError("The uploaded PDF did not produce any searchable chunks.")
    return chunks


def index_uploaded_pdf(file_storage: Any) -> dict[str, Any]:
    ensure_storage_directories()

    filename = getattr(file_storage, "filename", None) or getattr(file_storage, "name", None)
    if file_storage is None or not filename:
        raise ValueError("Please choose a PDF file to upload.")

    filename = sanitize_filename(filename)
    upload_stream = getattr(file_storage, "stream", file_storage)
    upload_stream.seek(0, os.SEEK_END)
    content_length = upload_stream.tell()
    upload_stream.seek(0)

    if content_length > settings.MAX_UPLOAD_SIZE:
        raise ValueError("The uploaded file exceeds the maximum allowed size.")

    file_content = upload_stream.read()
    document_hash = compute_document_hash(file_content)
    document_id = document_hash

    if document_already_indexed(document_hash):
        return {
            "success": False,
            "duplicate": True,
            "message": "This document has already been added to the collection.",
            "document_id": document_id,
            "filename": filename,
            "chunks_added": 0,
        }

    safe_path = Path(settings.UPLOAD_DIRECTORY) / filename
    safe_path.write_bytes(file_content)

    pages = extract_pdf_pages(str(safe_path))
    chunks = split_pages_into_chunks(pages, document_id, filename)

    vector_store = initialize_vector_store()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"{metadata['chunk_id']}" for metadata in metadatas]
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return {
        "success": True,
        "duplicate": False,
        "message": f"Indexed {filename} successfully.",
        "document_id": document_id,
        "filename": filename,
        "chunks_added": len(chunks),
    }


def clear_document_collection() -> dict[str, Any]:
    vector_store = initialize_vector_store()
    try:
        vector_store.delete_collection()
    except Exception:
        return {"success": False, "message": "The collection could not be cleared cleanly."}
    return {"success": True, "message": "The document collection was cleared."}
