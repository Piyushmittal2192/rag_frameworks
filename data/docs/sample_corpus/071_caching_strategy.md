# Caching Strategy

RAG systems can cache embeddings, retrieval results, reranker scores, generated answers, and judge results. Caches should include invalidation rules when documents, indexes, prompts, models, or user permissions change. Per-user memory should be considered when caching personalized answers.

