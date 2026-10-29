from __future__ import annotations

from typing import Any

import streamlit as st

from config import settings
from rag.graph import compiled_graph
from rag.ingestion import clear_document_collection, ensure_storage_directories, index_uploaded_pdf
from rag.retrieval import list_indexed_documents


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    st.subheader("Sources")
    for source in sources:
        filename = source.get("filename", "Unknown source")
        page_number = source.get("page_number")
        title = f"{filename} — page {page_number}" if page_number else filename
        with st.expander(title):
            st.write(source.get("excerpt", ""))


def main() -> None:
    st.set_page_config(page_title="Epilogue", page_icon="📚", layout="wide")
    ensure_storage_directories()

    st.title("Epilogue")
    st.caption("Evidence-grounded document intelligence for technical PDFs. Inference runs locally.")

    with st.sidebar:
        st.header("Upload PDFs")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Select PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help=f"Each PDF must be no larger than {settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
            )
            upload_submitted = st.form_submit_button("Upload and index")

        if upload_submitted:
            if not uploaded_files:
                st.warning("Please choose at least one PDF to upload.")
            else:
                for uploaded_file in uploaded_files:
                    try:
                        result = index_uploaded_pdf(uploaded_file)
                    except ValueError as exc:
                        st.error(f"{uploaded_file.name}: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected upload error for {uploaded_file.name}: {exc}")
                    else:
                        if result.get("success"):
                            st.success(result["message"])
                        elif result.get("duplicate"):
                            st.info(result["message"])
                        else:
                            st.warning(result.get("message", "The document was not indexed."))

        st.divider()
        st.header("Indexed documents")
        try:
            documents = list_indexed_documents()
        except Exception as exc:
            documents = []
            st.error(f"Could not load indexed documents: {exc}")

        if documents:
            for document in documents:
                with st.expander(document["filename"]):
                    st.caption(f"{document['chunk_count']} chunks")
                    st.write(f"Pages: {', '.join(map(str, document['page_numbers']))}")
        else:
            st.caption("No documents are indexed yet.")

        st.divider()
        st.header("Collection control")
        confirm_clear = st.checkbox("I understand this removes all indexed chunks")
        if st.button("Clear collection", type="secondary", disabled=not confirm_clear):
            try:
                result = clear_document_collection()
            except Exception as exc:
                st.error(f"The collection could not be cleared: {exc}")
            else:
                if result.get("success"):
                    st.success(result.get("message", "The document collection was cleared."))
                    st.rerun()
                else:
                    st.error(result.get("message", "The collection could not be cleared."))

    st.header("Ask a question")
    with st.form("question_form"):
        question = st.text_area(
            "Question",
            placeholder="Ask about the contents of your uploaded documents.",
        )
        question_submitted = st.form_submit_button("Submit question", type="primary")

    if question_submitted:
        cleaned_question = question.strip()
        if not cleaned_question:
            st.session_state.pop("question_result", None)
            st.error("Please enter a question before submitting.")
        else:
            initial_state = {
                "original_question": cleaned_question,
                "question": cleaned_question,
                "documents": [],
                "retry_count": 0,
                "answer": "",
                "sources": [],
                "sufficient_context": False,
                "workflow_status": "Question submitted.",
            }
            try:
                with st.spinner("Searching your documents and preparing an answer..."):
                    st.session_state["question_result"] = compiled_graph.invoke(initial_state)
            except Exception as exc:
                st.session_state.pop("question_result", None)
                st.error(f"The question workflow failed: {exc}")

    result = st.session_state.get("question_result")
    if result:
        st.subheader("Answer")
        st.markdown(f"**Question:** {result.get('original_question', '')}")
        st.write(
            result.get("answer")
            or "The available evidence is insufficient to support an answer."
        )
        st.caption(
            f"{result.get('workflow_status', 'Question processed.')} "
            f"Retries: {result.get('retry_count', 0)}"
        )
        _render_sources(result.get("sources") or [])


if __name__ == "__main__":
    main()
