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


class PersonalizationContext(BaseModel):
    mode: str = "stateless"
    user_id: str | None = None
    preferences: dict[str, str] = Field(default_factory=dict)
    persisted_preferences: dict[str, str] = Field(default_factory=dict)
    session_preferences: dict[str, str] = Field(default_factory=dict)
    conversation_summary: str | None = None
    current_goal: str | None = None
    user_intent: str | None = None
    recent_topics: list[str] = Field(default_factory=list)
    scratchpad_facts: list[str] = Field(default_factory=list)
    scratchpad_decisions: list[str] = Field(default_factory=list)
    scratchpad_open_questions: list[str] = Field(default_factory=list)
    memory_loaded: bool = False
    memory_saved: bool = False
    conversation_memory_loaded: bool = False
    conversation_memory_saved: bool = False
    scratchpad_memory_loaded: bool = False
    scratchpad_memory_saved: bool = False


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
    judge_enabled: bool = False
    judge_model: str | None = None
    judge_verdict: str | None = None
    faithfulness_score: float | None = None
    memory_mode: str = "stateless"
    memory_user_id: str | None = None
    memory_loaded: bool = False
    memory_saved: bool = False
    personalization_preferences: list[str] = Field(default_factory=list)
    conversation_memory_loaded: bool = False
    conversation_memory_saved: bool = False
    scratchpad_memory_loaded: bool = False
    scratchpad_memory_saved: bool = False
    recent_topics: list[str] = Field(default_factory=list)
    scratchpad_items: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    verdict: str
    faithfulness_score: float
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    reason: str


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]
    pipeline: str
    steps: list[PipelineStep] = Field(default_factory=list)
    trace: TraceMetadata | None = None
    judge: JudgeResult | None = None
    personalization: PersonalizationContext | None = None
    rewritten_query: str | None = None
    correction_applied: bool = False
