# Access Control

Access control should happen before answer generation. Retrieval should filter candidate documents by user permissions. The LLM should not receive chunks the user is not allowed to see. Trace logs should avoid leaking restricted document text to unauthorized users.

