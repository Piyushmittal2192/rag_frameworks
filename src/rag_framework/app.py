from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_framework.config import get_settings
from rag_framework.embeddings import SentenceTransformerEmbedder
from rag_framework.judges import LLMFaithfulnessJudge
from rag_framework.llms import create_llm
from rag_framework.models import TraceMetadata
from rag_framework.pipelines import CorrectiveRAGPipeline, PlannerRAGPipeline, StandardRAGPipeline
from rag_framework.rerankers import CrossEncoderReranker
from rag_framework.vector_store import VectorStore


class QueryRequest(BaseModel):
    question: str
    pipeline: Literal["standard", "corrective", "planner"] = "standard"
    top_k: int | None = None


class AppState:
    standard: StandardRAGPipeline
    corrective: CorrectiveRAGPipeline
    planner: PlannerRAGPipeline
    llm_provider: str
    llm_model: str
    reranker_enabled: bool
    reranker_model: str | None
    judge_enabled: bool
    judge_model: str | None


state = AppState()
STATIC_DIR = Path(__file__).with_name("static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore.load(settings.index_dir)
    llm = create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or settings.github_token,
    )
    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.enable_reranker
        else None
    )
    judge_llm = None
    if settings.enable_llm_judge:
        judge_llm = create_llm(
            provider=settings.judge_provider or settings.llm_provider,
            model=settings.judge_model or settings.llm_model,
            base_url=settings.judge_base_url or settings.llm_base_url,
            api_key=settings.judge_api_key or settings.llm_api_key or settings.github_token,
        )
    judge = LLMFaithfulnessJudge(judge_llm) if judge_llm is not None else None
    state.standard = StandardRAGPipeline(
        store,
        embedder,
        llm,
        top_k=settings.top_k,
        reranker=reranker,
        judge=judge,
    )
    state.corrective = CorrectiveRAGPipeline(
        store,
        embedder,
        llm,
        top_k=settings.top_k,
        relevance_threshold=settings.relevance_threshold,
        rewrite_intent_threshold=settings.rewrite_intent_threshold,
        rewrite_evidence_margin=settings.rewrite_evidence_margin,
        reranker_evidence_threshold=settings.reranker_evidence_threshold,
        reranker=reranker,
        judge=judge,
    )
    state.planner = PlannerRAGPipeline(
        store,
        embedder,
        llm,
        top_k=settings.top_k,
        reranker=reranker,
        judge=judge,
    )
    state.llm_provider = settings.llm_provider
    state.llm_model = settings.llm_model
    state.reranker_enabled = settings.enable_reranker
    state.reranker_model = settings.reranker_model if settings.enable_reranker else None
    state.judge_enabled = settings.enable_llm_judge
    state.judge_model = (
        (settings.judge_model or settings.llm_model)
        if settings.enable_llm_judge
        else None
    )
    yield


app = FastAPI(title="RAG Framework", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest):
    pipelines = {
        "standard": state.standard,
        "corrective": state.corrective,
        "planner": state.planner,
    }
    pipeline = pipelines[request.pipeline]
    original_top_k = pipeline.top_k
    if request.top_k is not None:
        pipeline.top_k = request.top_k
    started_at = datetime.now(timezone.utc)
    timer = perf_counter()
    try:
        answer = await pipeline.answer(request.question)
        answer.trace = TraceMetadata(
            trace_id=str(uuid4()),
            started_at=started_at.isoformat(),
            total_duration_ms=round((perf_counter() - timer) * 1000, 2),
            llm_provider=state.llm_provider,
            llm_model=state.llm_model,
            top_k=pipeline.top_k,
            source_count=len(answer.sources),
            correction_applied=answer.correction_applied,
            best_score=max((_relevance_score(source) for source in answer.sources), default=0.0),
            mean_score=(
                sum(_relevance_score(source) for source in answer.sources) / len(answer.sources)
                if answer.sources
                else 0.0
            ),
            reranker_enabled=state.reranker_enabled,
            reranker_model=state.reranker_model,
            judge_enabled=state.judge_enabled,
            judge_model=state.judge_model,
            judge_verdict=answer.judge.verdict if answer.judge else None,
            faithfulness_score=answer.judge.faithfulness_score if answer.judge else None,
        )
        return answer
    finally:
        pipeline.top_k = original_top_k


def _relevance_score(source) -> float:
    return source.raw_scores.get("vector", source.score)
