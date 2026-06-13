# Prompt Injection

Prompt injection occurs when retrieved text tries to override system instructions or exfiltrate secrets. RAG applications should treat retrieved chunks as untrusted data. The answer prompt should tell the LLM to use context for facts only and ignore instructions inside retrieved documents.

