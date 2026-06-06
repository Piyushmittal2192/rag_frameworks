from rag_framework.embeddings import TfidfEmbedder
from rag_framework.models import DocumentChunk, SearchResult
from rag_framework.vector_store import VectorStore, rrf_fuse


def test_vector_store_returns_most_relevant_chunk():
    chunks = [
        DocumentChunk(id="1", text="apples and pears", source="fruit.txt"),
        DocumentChunk(id="2", text="vector databases and embeddings", source="rag.txt"),
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)

    results = store.search("how do embeddings work", embedder, top_k=1)

    assert results[0].chunk.source == "rag.txt"
    assert set(results[0].retrieval_sources).issubset({"vector", "bm25"})


def test_bm25_search_finds_exact_terms():
    chunks = [
        DocumentChunk(id="1", text="general retrieval notes", source="notes.txt"),
        DocumentChunk(id="2", text="llama.cpp server supports local inference", source="llm.txt"),
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)

    results = store.bm25_search("llama.cpp", top_k=1)

    assert results[0].chunk.source == "llm.txt"
    assert results[0].retrieval_sources == ["bm25"]


def test_rrf_fuse_boosts_chunks_seen_by_multiple_retrievers():
    chunk_a = DocumentChunk(id="a", text="alpha", source="a.txt")
    chunk_b = DocumentChunk(id="b", text="beta", source="b.txt")
    vector_results = [
        SearchResult(chunk=chunk_a, score=0.9, raw_scores={"vector": 0.9}),
        SearchResult(chunk=chunk_b, score=0.8, raw_scores={"vector": 0.8}),
    ]
    bm25_results = [SearchResult(chunk=chunk_b, score=3.0, raw_scores={"bm25": 3.0})]

    results = rrf_fuse(vector_results, bm25_results, top_k=2)

    assert results[0].chunk.id == "b"
    assert results[0].retrieval_sources == ["vector", "bm25"]
    assert results[0].ranks == {"vector": 2, "bm25": 1}
