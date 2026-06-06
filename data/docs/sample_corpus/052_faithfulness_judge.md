# Faithfulness Judge

The framework can run an optional secondary LLM-as-judge after answer generation. The judge receives the original question, retrieved context, and generated answer. It returns a verdict such as grounded, partially grounded, unsupported, or judge error, plus a faithfulness score, unsupported claims, citation issues, and a short reason. The judge is designed to make hallucination risk visible in the UI and trace, but it should be combined with retrieval evaluation and curated test datasets for production trust.
