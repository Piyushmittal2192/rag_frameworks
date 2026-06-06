from typing import Protocol

from sentence_transformers import CrossEncoder

from rag_framework.models import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, result.chunk.text) for result in results]
        scores = self.model.predict(pairs)
        scored_results = []
        for result, score in zip(results, scores, strict=True):
            rerank_score = float(score)
            result.raw_scores["reranker"] = rerank_score
            scored_results.append(result)

        scored_results.sort(key=lambda result: result.raw_scores["reranker"], reverse=True)
        for rank, result in enumerate(scored_results, start=1):
            result.ranks["reranker"] = rank
            result.score = result.raw_scores["reranker"]
        return scored_results[:top_k]
