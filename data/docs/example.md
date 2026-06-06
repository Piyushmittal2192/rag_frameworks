# Example Knowledge Base

This framework demonstrates two retrieval augmented generation pipelines.

Standard RAG retrieves relevant document chunks and asks an LLM to answer from those chunks.

Corrective RAG checks whether retrieved chunks are relevant. If the evidence is weak, it rewrites the search query, retrieves again, filters the context, and then answers.

The project supports Ollama and OpenAI-compatible local model servers such as vLLM, llama.cpp server, and LM Studio.
