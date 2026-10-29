from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import settings
from rag.prompts import ANSWER_GENERATION_PROMPT, QUERY_REWRITE_PROMPT, RELEVANCE_GRADE_PROMPT
from rag.retrieval import build_source_references, retrieve_documents
from rag.state import GraphState


class OllamaChatWrapper:
    def __init__(self, model_name: str, base_url: str) -> None:
        self._chat_model = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.2,
            num_gpu=settings.OLLAMA_NUM_GPU,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=settings.OLLAMA_NUM_PREDICT,
            reasoning=False,
        )

    def invoke(self, prompt: str):
        return self._chat_model.invoke(prompt)


llm = OllamaChatWrapper(
    model_name=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)


def parse_grade_response(raw_response: str) -> dict[str, object]:
    cleaned = raw_response.strip()
    if not cleaned:
        return {"relevant": False, "explanation": "The model returned an empty relevance score."}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    lower_response = cleaned.lower()
    relevant = "true" in lower_response or "relevant" in lower_response and "not" not in lower_response
    return {"relevant": relevant, "explanation": cleaned[:200]}


def retrieve_documents_node(state: GraphState) -> GraphState:
    question = state.get("question") or state.get("original_question") or ""
    documents = retrieve_documents(question)
    state["documents"] = documents
    state["workflow_status"] = "Retrieved candidate passages from the local document index."
    return state


def grade_documents_node(state: GraphState) -> GraphState:
    documents = state.get("documents") or []
    question = state.get("question") or state.get("original_question")

    if not documents:
        state["sufficient_context"] = False
        state["answer"] = "The available documents do not contain enough evidence to answer this question."
        state["sources"] = []
        state["workflow_status"] = "No document chunks were retrieved for this question."
        return state

    context_text = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source_filename', 'unknown')} | Page: {doc.metadata.get('page_number', 'unknown')}\n{doc.page_content[:800]}"
        for doc in documents
    )
    prompt = RELEVANCE_GRADE_PROMPT.format(question=question, context=context_text)
    result = llm.invoke(prompt)
    grade = parse_grade_response(str(result.content))
    relevant = bool(grade.get("relevant", False))
    explanation = str(grade.get("explanation", "The passages were not considered sufficiently relevant."))

    if relevant:
        state["sufficient_context"] = True
        state["workflow_status"] = "Relevant evidence was found and can support an answer."
        return state

    if state.get("retry_count", 0) >= settings.MAX_RETRIES:
        state["sufficient_context"] = False
        state["answer"] = "The available documents do not provide enough evidence to answer this question reliably."
        state["sources"] = []
        state["workflow_status"] = (
            "The evidence remained insufficient after the retry limit. "
            f"Reason: {explanation}"
        )
        return state

    state["sufficient_context"] = False
    state["workflow_status"] = f"The retrieved passages were weakly relevant: {explanation}"
    return state


def rewrite_question_node(state: GraphState) -> GraphState:
    original_question = state.get("original_question")
    documents = state.get("documents") or []
    context = "\n\n".join(doc.page_content[:500] for doc in documents)
    prompt = QUERY_REWRITE_PROMPT.format(original_question=original_question, context=context)
    response = llm.invoke(prompt)
    rewritten_question = str(response.content).strip() or original_question
    state["question"] = rewritten_question
    state["retry_count"] = int(state.get("retry_count", 0)) + 1
    state["workflow_status"] = (
        f"Query rewriting triggered retry {state['retry_count']}/{settings.MAX_RETRIES}."
    )
    return state


def generate_answer_node(state: GraphState) -> GraphState:
    documents = state.get("documents") or []
    question = state.get("question") or state.get("original_question")
    if not documents:
        state["answer"] = "The available documents are insufficient to answer this question."
        state["sources"] = []
        state["sufficient_context"] = False
        state["workflow_status"] = "No supporting material was available for answer generation."
        return state

    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source_filename', 'unknown')} | Page: {doc.metadata.get('page_number', 'unknown')}\n{doc.page_content[:1200]}"
        for doc in documents
    )
    prompt = ANSWER_GENERATION_PROMPT.format(question=question, context=context)
    response = llm.invoke(prompt)
    answer = str(response.content).strip()
    effective_sources = build_source_references(documents)

    state["answer"] = answer or "The available evidence does not support a confident answer."
    state["sources"] = effective_sources
    state["workflow_status"] = f"Generated an answer using {len(effective_sources)} source reference(s)."
    state["sufficient_context"] = True
    return state


def route_after_grade(state: GraphState) -> str:
    if state.get("sufficient_context"):
        return "generate_answer"
    if state.get("retry_count", 0) >= settings.MAX_RETRIES:
        return END
    return "rewrite_question"


def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
            END: END,
        },
    )
    workflow.add_edge("rewrite_question", "retrieve_documents")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


compiled_graph = build_graph()
