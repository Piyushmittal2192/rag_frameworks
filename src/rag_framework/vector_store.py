import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rag_framework.embeddings import Embedder
from rag_framework.models import DocumentChunk, SearchResult
from rag_framework.rerankers import Reranker


RRF_K = 60


class VectorStore:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
        bm25_index: "BM25Index | None" = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.bm25_index = bm25_index or BM25Index.build(chunks)

    @classmethod
    def build(cls, chunks: list[DocumentChunk], embedder: Embedder) -> "VectorStore":
        vectors = embedder.embed([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, bm25_index=BM25Index.build(chunks))

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.jsonl").write_text(
            "\n".join(chunk.model_dump_json() for chunk in self.chunks),
            encoding="utf-8",
        )
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "bm25.json").write_text(
            json.dumps(self.bm25_index.to_json(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        chunk_path = index_dir / "chunks.jsonl"
        vector_path = index_dir / "vectors.npy"
        if not chunk_path.exists() or not vector_path.exists():
            raise FileNotFoundError(f"No index found in {index_dir}. Run ingestion first.")
        chunks = [
            DocumentChunk.model_validate(json.loads(line))
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(vector_path)
        bm25_path = index_dir / "bm25.json"
        bm25_index = (
            BM25Index.from_json(json.loads(bm25_path.read_text(encoding="utf-8")))
            if bm25_path.exists()
            else BM25Index.build(chunks)
        )
        return cls(chunks=chunks, vectors=vectors, bm25_index=bm25_index)

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int,
        reranker: Reranker | None = None,
    ) -> list[SearchResult]:
        candidate_k = max(top_k * 4, top_k)
        vector_results = self.vector_search(query, embedder, candidate_k)
        bm25_results = self.bm25_search(query, candidate_k)
        fused = rrf_fuse(vector_results, bm25_results, top_k=candidate_k)
        if reranker is not None:
            return reranker.rerank(query, fused, top_k=top_k)
        return fused[:top_k]

    def vector_search(self, query: str, embedder: Embedder, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        query_vector = embedder.embed([query])[0]
        scores = self.vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                retrieval_sources=["vector"],
                ranks={"vector": rank},
                raw_scores={"vector": float(scores[index])},
            )
            for rank, index in enumerate(top_indices, start=1)
        ]

    def bm25_search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        ranked = self.bm25_index.search(query, top_k)
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=score,
                retrieval_sources=["bm25"],
                ranks={"bm25": rank},
                raw_scores={"bm25": score},
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


class BM25Index:
    def __init__(
        self,
        tokenized_documents: list[list[str]],
        idf: dict[str, float],
        avg_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenized_documents = tokenized_documents
        self.idf = idf
        self.avg_document_length = avg_document_length
        self.k1 = k1
        self.b = b
        self.document_frequencies = [Counter(document) for document in tokenized_documents]

    @classmethod
    def build(cls, chunks: list[DocumentChunk]) -> "BM25Index":
        tokenized_documents = [_tokenize(chunk.text) for chunk in chunks]
        document_count = len(tokenized_documents)
        document_occurrences: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_occurrences.update(set(tokens))

        idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_occurrences.items()
        }
        avg_document_length = (
            sum(len(document) for document in tokenized_documents) / document_count
            if document_count
            else 0.0
        )
        return cls(
            tokenized_documents=tokenized_documents,
            idf=idf,
            avg_document_length=avg_document_length,
        )

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores = []
        for index, document in enumerate(self.tokenized_documents):
            score = self._score_document(query_terms, index, len(document))
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def to_json(self) -> dict[str, object]:
        return {
            "tokenized_documents": self.tokenized_documents,
            "idf": self.idf,
            "avg_document_length": self.avg_document_length,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "BM25Index":
        return cls(
            tokenized_documents=[
                list(document) for document in payload["tokenized_documents"]  # type: ignore[index]
            ],
            idf={str(key): float(value) for key, value in dict(payload["idf"]).items()},
            avg_document_length=float(payload["avg_document_length"]),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def _score_document(self, query_terms: list[str], document_index: int, document_length: int) -> float:
        frequencies = self.document_frequencies[document_index]
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.avg_document_length, 1e-9)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    rrf_scores: defaultdict[str, float] = defaultdict(float)

    for source_name, results in (("vector", vector_results), ("bm25", bm25_results)):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            rrf_scores[chunk_id] += 1 / (rrf_k + rank)
            if chunk_id not in fused:
                fused[chunk_id] = SearchResult(chunk=result.chunk, score=0.0)
            fused_result = fused[chunk_id]
            if source_name not in fused_result.retrieval_sources:
                fused_result.retrieval_sources.append(source_name)
            fused_result.ranks[source_name] = rank
            fused_result.raw_scores[source_name] = result.raw_scores.get(source_name, result.score)

    ranked = sorted(fused.values(), key=lambda result: rrf_scores[result.chunk.id], reverse=True)
    for result in ranked:
        score = float(rrf_scores[result.chunk.id])
        result.score = score
        result.raw_scores["rrf"] = score
    return ranked[:top_k]


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]*", text.lower())
        if len(token) > 1
    ]
