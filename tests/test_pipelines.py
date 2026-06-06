import pytest

from rag_framework.embeddings import TfidfEmbedder
from rag_framework.llms import EchoLLM
from rag_framework.models import DocumentChunk
from rag_framework.pipelines import CorrectiveRAGPipeline, PlannerRAGPipeline, StandardRAGPipeline
from rag_framework.vector_store import VectorStore


class RewriteLLM:
    async def generate(self, prompt: str) -> str:
        if "Rewrite the question" in prompt:
            return "semantic search embeddings vector databases"
        if "Decide whether" in prompt:
            return "relevant" if "embeddings" in prompt else "irrelevant"
        return "final answer"


class PlannerLLM:
    async def generate(self, prompt: str) -> str:
        if "Break the question into focused search sub-questions" in prompt:
            return "1. What is vector retrieval?\n2. What is BM25 retrieval?"
        return "planned final answer"


@pytest.mark.asyncio
async def test_standard_pipeline_returns_sources():
    chunks = [DocumentChunk(id="1", text="RAG uses retrieved context.", source="rag.txt")]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = StandardRAGPipeline(store, embedder, EchoLLM(), top_k=1)

    answer = await pipeline.answer("What does RAG use?")

    assert answer.pipeline == "standard"
    assert answer.sources[0].chunk.source == "rag.txt"
    assert "Echo response" not in answer.answer


@pytest.mark.asyncio
async def test_demo_llm_does_not_infer_unsupported_items_from_positive_context():
    chunks = [
        DocumentChunk(
            id="1",
            text="The project supports Ollama and OpenAI-compatible local model servers.",
            source="rag.txt",
        )
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = StandardRAGPipeline(store, embedder, EchoLLM(), top_k=1)

    answer = await pipeline.answer("What open source LLM interfaces are not supported?")

    assert "does not state" in answer.answer


@pytest.mark.asyncio
async def test_corrective_pipeline_rewrites_when_scores_are_weak():
    chunks = [
        DocumentChunk(id="1", text="unrelated cooking notes", source="cooking.txt"),
        DocumentChunk(id="2", text="embeddings power vector databases", source="rag.txt"),
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = CorrectiveRAGPipeline(
        store,
        embedder,
        RewriteLLM(),
        top_k=1,
        relevance_threshold=0.9,
    )

    answer = await pipeline.answer("Tell me about semantic search")

    assert answer.pipeline == "corrective"
    assert answer.correction_applied is True
    assert answer.rewritten_query == "semantic search embeddings vector databases"


@pytest.mark.asyncio
async def test_corrective_pipeline_rejects_rewrite_when_intent_drifts():
    class DriftLLM:
        async def generate(self, prompt: str) -> str:
            if "Rewrite the question" in prompt:
                return "open source LLM interfaces supported"
            if "Decide whether" in prompt:
                return "irrelevant"
            return "final answer"

    chunks = [
        DocumentChunk(id="1", text="Ollama and vLLM are supported interfaces.", source="rag.txt")
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = CorrectiveRAGPipeline(
        store,
        embedder,
        DriftLLM(),
        top_k=1,
        relevance_threshold=0.9,
    )

    answer = await pipeline.answer("What open source LLM interfaces are not supported?")

    assert answer.correction_applied is False
    assert any(step.name == "Rewrite Decision" and step.status == "rejected" for step in answer.steps)


@pytest.mark.asyncio
async def test_corrective_pipeline_triggers_when_reranker_evidence_is_weak():
    class StaticReranker:
        def rerank(self, query, results, top_k):
            for rank, result in enumerate(results, start=1):
                result.raw_scores["reranker"] = -2.0
                result.ranks["reranker"] = rank
                result.score = -2.0
            return results[:top_k]

    chunks = [
        DocumentChunk(id="1", text="semantic search embeddings vector databases", source="rag.txt"),
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = CorrectiveRAGPipeline(
        store,
        embedder,
        RewriteLLM(),
        top_k=1,
        relevance_threshold=0.0,
        reranker_evidence_threshold=0.0,
        reranker=StaticReranker(),
    )

    answer = await pipeline.answer("Tell me about semantic search")

    grade_step = next(step for step in answer.steps if step.name == "Grade Context")
    assert grade_step.details["weak_evidence"] is True
    assert any(step.name == "Rewrite Query" for step in answer.steps)


@pytest.mark.asyncio
async def test_planner_pipeline_decomposes_and_fuses_subquery_sources():
    chunks = [
        DocumentChunk(id="1", text="Vector retrieval uses embeddings.", source="vector.txt"),
        DocumentChunk(id="2", text="BM25 retrieval uses lexical term matching.", source="bm25.txt"),
        DocumentChunk(id="3", text="Docker containers package the service.", source="docker.txt"),
    ]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = PlannerRAGPipeline(store, embedder, PlannerLLM(), top_k=2)

    answer = await pipeline.answer("Compare vector retrieval and BM25 retrieval.")

    assert answer.pipeline == "planner"
    assert answer.answer == "planned final answer"
    assert {source.chunk.source for source in answer.sources} == {"vector.txt", "bm25.txt"}
    assert any(step.name == "Decompose Query" for step in answer.steps)
    assert any(step.name == "Fuse Sub-query Evidence" for step in answer.steps)


@pytest.mark.asyncio
async def test_planner_pipeline_keeps_single_question_when_plan_has_one_line():
    chunks = [DocumentChunk(id="1", text="RAG uses retrieved context.", source="rag.txt")]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    pipeline = PlannerRAGPipeline(store, embedder, EchoLLM(), top_k=1)

    answer = await pipeline.answer("What does RAG use?")

    plan_step = next(step for step in answer.steps if step.name == "Decompose Query")
    assert plan_step.details["sub_question_count"] == 1
    assert plan_step.details["sub_questions"] == ["What does RAG use?"]
