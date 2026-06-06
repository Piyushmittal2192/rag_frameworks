ANSWER_PROMPT = """You are a careful retrieval-augmented assistant.
Answer the question using only the provided context. If the context is insufficient, say so clearly.
Do not infer missing facts from nearby facts. If the question asks what is not supported,
only answer with unsupported items that are explicitly named in the context.
When the context contains an explicit list, preserve every item in that list.

Question:
{question}

Context:
{context}

Answer with a concise explanation and cite sources as [source: chunk].
"""

REWRITE_PROMPT = """Rewrite the question into a better search query for retrieving factual context.
Keep proper nouns, expand acronyms only when obvious, and return one standalone query.

Question: {question}
Search query:"""

GRADE_PROMPT = """Decide whether the context is relevant to the question.
Return only "relevant" or "irrelevant".

Question:
{question}

Context:
{context}
"""

DECOMPOSE_PROMPT = """You are planning retrieval for a RAG system.
Break the question into focused search sub-questions only when that improves retrieval.
Use the original wording for important entities, products, and technical terms.
If the question is already a single focused lookup, return exactly one line with the original question.
Return at most {max_sub_questions} numbered lines. Do not answer the question.

Question:
{question}

Sub-questions:
"""

JUDGE_PROMPT = """You are a strict RAG faithfulness judge.
Decide whether the generated answer is fully supported by the retrieved context.

Rules:
- Judge only factual support from the provided context.
- Do not use outside knowledge.
- Penalize claims that are not directly supported by the context.
- Penalize citations that do not support the cited claim.
- If the context is insufficient and the answer still makes factual claims, mark it unsupported.
- Return strict JSON only, with no markdown.

Valid verdicts:
- grounded
- partially_grounded
- unsupported

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}

Return JSON with this schema:
{{
  "verdict": "grounded",
  "faithfulness_score": 1.0,
  "unsupported_claims": [],
  "citation_issues": [],
  "reason": "Short explanation."
}}
"""
