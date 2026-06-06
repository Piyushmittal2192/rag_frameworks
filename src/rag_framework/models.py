from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    retrieval_sources: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    raw_scores: dict[str, float] = Field(default_factory=dict)


class PipelineStep(BaseModel):
    name: str
    status: str
    description: str
    duration_ms: float = 0.0
    details: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class TraceMetadata(BaseModel):
    trace_id: str
    started_at: str
    total_duration_ms: float
    llm_provider: str
    llm_model: str
    top_k: int
    source_count: int
    correction_applied: bool
    best_score: float
    mean_score: float
    reranker_enabled: bool = False
    reranker_model: str | None = None


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]
    pipeline: str
    steps: list[PipelineStep] = Field(default_factory=list)
    trace: TraceMetadata | None = None
    rewritten_query: str | None = None
    correction_applied: bool = False
