# Planner RAG

Planner RAG adds dynamic task decomposition for broad or multi-part questions. It asks the LLM to create focused sub-questions, retrieves evidence for each sub-question with hybrid vector and BM25 search, fuses the evidence with rank-based scoring, optionally reranks the combined candidates, and synthesizes one cited answer. Planner RAG is best for comparison, analysis, and multi-topic questions where one retrieval query may miss important evidence.
