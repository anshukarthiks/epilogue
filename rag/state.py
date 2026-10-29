from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    original_question: str
    question: str
    documents: list[Document]
    retry_count: int
    answer: str
    sources: list[dict]
    sufficient_context: bool
    workflow_status: str
