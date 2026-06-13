# Multi-tenant RAG

Multi-tenant RAG must isolate documents, indexes, memory, and traces by tenant. A user from one tenant should never retrieve another tenant's documents or stateful memory. Metadata filters, access checks, and audit logs are required before retrieval results are sent to an LLM.

