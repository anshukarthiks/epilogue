from __future__ import annotations

import json
from pathlib import Path

from rag.retrieval import retrieve_documents


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "sample_questions.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    for entry in dataset:
        question = entry["question"]
        relevant = retrieve_documents(question, top_k=4)
        print(f"Question: {question}")
        print(f"Retrieved chunks: {len(relevant)}")
        for document in relevant[:3]:
            print("-", document.page_content[:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
