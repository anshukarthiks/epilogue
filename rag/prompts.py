RELEVANCE_GRADE_PROMPT = """
You are checking whether retrieved document passages are relevant to a question.
Return valid JSON only with two keys:
- relevant: true or false
- explanation: a short, plain-language explanation

Question:
{question}

Retrieved passages:
{context}

Evaluate whether the passages contain usable evidence for the question.
If the texts are unrelated, too weak, or insufficient, set relevant to false.
Do not claim certainty that the answer is complete or correct.
"""

QUERY_REWRITE_PROMPT = """
Rewrite the search question to improve retrieval using the available evidence.
Keep the original meaning and do not add unsupported assumptions.
Return only the rewritten question.

Original question:
{original_question}

Evidence context:
{context}
"""

ANSWER_GENERATION_PROMPT = """
Use the supplied document excerpts as the evidence base for your answer.
Answer the user's question with the strongest supported claim available.
If the evidence is insufficient, say so clearly and avoid inventing details.
Differentiate between information directly supported by the documents and reasonable inferences.
Include source references by using the metadata from the supplied passages.

User question:
{question}

Evidence:
{context}
"""
