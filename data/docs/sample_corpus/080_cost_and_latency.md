# Cost And Latency

RAG latency comes from embedding the query, searching indexes, reranking candidates, generating the answer, and optionally running a judge. Cost can increase with planner decomposition, corrective retries, larger context windows, and hosted model calls. Observability should report each stage separately.

