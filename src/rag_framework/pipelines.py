import re
from time import perf_counter

import numpy as np

from rag_framework.embeddings import Embedder
from rag_framework.judges import Judge
from rag_framework.llms import LLM
from rag_framework.memory import format_personalization_context, summarize_preferences
from rag_framework.models import Answer, PersonalizationContext, PipelineStep, SearchResult
from rag_framework.prompts import ANSWER_PROMPT, DECOMPOSE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
from rag_framework.rerankers import Reranker
from rag_framework.vector_store import VectorStore


class StandardRAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        top_k: int = 5,
        reranker: Reranker | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.reranker = reranker
        self.judge = judge

    async def answer(
        self,
        question: str,
        personalization: PersonalizationContext | None = None,
    ) -> Answer:
        started = perf_counter()
        results = self.store.search(question, self.embedder, self.top_k, self.reranker)
        retrieve_ms = _elapsed_ms(started)

        started = perf_counter()
        prompt = _build_answer_prompt(question, results, personalization)
        build_ms = _elapsed_ms(started)

        started = perf_counter()
        response = await self.llm.generate(prompt)
        generate_ms = _elapsed_ms(started)

        answer = Answer(
            question=question,
            answer=response,
            sources=results,
            pipeline="standard",
            steps=[
                PipelineStep(
                    name="Retrieve",
                    status="completed",
                    description="Embedded the question and searched the vector index for matching chunks.",
                    duration_ms=retrieve_ms,
                    details={
                        "query": question,
                        "top_k": self.top_k,
                        "matches": len(results),
                        "best_score": _best_score(results),
                        "reranker": self.reranker is not None,
                    },
                ),
                PipelineStep(
                    name="Build Context",
                    status="completed",
                    description="Packed the retrieved chunks into the answer prompt.",
                    duration_ms=build_ms,
                    details={
                        "sources": _source_labels(results),
                        **_personalization_details(personalization),
                    },
                ),
                PipelineStep(
                    name="Generate Answer",
                    status="completed",
                    description="Sent the question and retrieved context to the configured LLM interface.",
                    duration_ms=generate_ms,
                    details={"context_chunks": len(results)},
                ),
            ],
            personalization=personalization,
        )
        return await self._run_judge(answer)

    async def _run_judge(self, answer: Answer) -> Answer:
        if self.judge is None:
            answer.steps.append(
                PipelineStep(
                    name="LLM Judge",
                    status="skipped",
                    description="Faithfulness judge is disabled for this run.",
                    details={"judge_enabled": False},
                )
            )
            return answer

        started = perf_counter()
        judge_result = await self.judge.evaluate(answer.question, answer.answer, answer.sources)
        judge_ms = _elapsed_ms(started)
        answer.judge = judge_result
        answer.steps.append(
            PipelineStep(
                name="LLM Judge",
                status="completed" if judge_result.verdict != "judge_error" else "warning",
                description="Checked the generated answer against the retrieved context.",
                duration_ms=judge_ms,
                details={
                    "verdict": judge_result.verdict,
                    "faithfulness_score": judge_result.faithfulness_score,
                    "unsupported_claims": judge_result.unsupported_claims or ["none"],
                    "citation_issues": judge_result.citation_issues or ["none"],
                },
            )
        )
        return answer


class CorrectiveRAGPipeline(StandardRAGPipeline):
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        top_k: int = 5,
        relevance_threshold: float = 0.28,
        rewrite_intent_threshold: float = 0.65,
        rewrite_evidence_margin: float = 0.05,
        reranker_evidence_threshold: float = 0.0,
        reranker: Reranker | None = None,
        judge: Judge | None = None,
    ) -> None:
        super().__init__(
            store=store,
            embedder=embedder,
            llm=llm,
            top_k=top_k,
            reranker=reranker,
            judge=judge,
        )
        self.relevance_threshold = relevance_threshold
        self.rewrite_intent_threshold = rewrite_intent_threshold
        self.rewrite_evidence_margin = rewrite_evidence_margin
        self.reranker_evidence_threshold = reranker_evidence_threshold

    async def answer(
        self,
        question: str,
        personalization: PersonalizationContext | None = None,
    ) -> Answer:
        started = perf_counter()
        initial_results = self.store.search(question, self.embedder, self.top_k, self.reranker)
        initial_retrieval_ms = _elapsed_ms(started)

        started = perf_counter()
        filtered = await self._filter_relevant(question, initial_results)
        grade_ms = _elapsed_ms(started)

        correction_applied = False
        rewritten_query: str | None = None
        initial_evidence_score = _evidence_score(filtered or initial_results)
        weak_evidence = self._is_weak_evidence(filtered, initial_evidence_score)
        trigger_reasons = self._evidence_trigger_reasons(filtered, initial_evidence_score)
        steps = [
            PipelineStep(
                name="Initial Retrieval",
                status="completed",
                description="Embedded the original question and retrieved the first candidate chunks.",
                duration_ms=initial_retrieval_ms,
                details={
                    "query": question,
                    "top_k": self.top_k,
                    "matches": len(initial_results),
                    "best_score": _best_score(initial_results),
                    "reranker": self.reranker is not None,
                },
            ),
            PipelineStep(
                name="Grade Context",
                status="completed",
                description="Checked whether retrieved chunks were strong enough to answer from.",
                duration_ms=grade_ms,
                details={
                    "kept_chunks": len(filtered),
                    "mean_score": _mean_score(filtered),
                    "threshold": self.relevance_threshold,
                    "evidence_score": initial_evidence_score,
                    "reranker_evidence_threshold": self.reranker_evidence_threshold,
                    "weak_evidence": weak_evidence,
                },
            ),
            PipelineStep(
                name="Correction Check",
                status="triggered" if weak_evidence else "skipped",
                description=(
                    "The retrieved context needs a rewrite attempt."
                    if weak_evidence
                    else "The initial retrieved context passed the correction gates."
                ),
                details={
                    "correction_triggered": weak_evidence,
                    "reasons": trigger_reasons,
                    "mean_score": _mean_score(filtered),
                    "threshold": self.relevance_threshold,
                    "evidence_score": initial_evidence_score,
                    "reranker_evidence_threshold": self.reranker_evidence_threshold,
                },
            ),
        ]

        if weak_evidence:
            started = perf_counter()
            rewritten_query = await self._rewrite_query(question)
            rewrite_quality = self._validate_rewrite(question, rewritten_query)
            rewrite_ms = _elapsed_ms(started)
            steps.append(
                PipelineStep(
                    name="Rewrite Query",
                    status="completed" if rewrite_quality["intent_preserved"] else "rejected",
                    description="Rewrote the question, then checked whether it preserved the original intent.",
                    duration_ms=rewrite_ms,
                    details={
                        "rewritten_query": rewritten_query,
                        "intent_similarity": rewrite_quality["intent_similarity"],
                        "term_preservation": rewrite_quality["term_preservation"],
                        "polarity_changed": rewrite_quality["polarity_changed"],
                        "intent_preserved": rewrite_quality["intent_preserved"],
                        "intent_threshold": self.rewrite_intent_threshold,
                    },
                )
            )

            if rewrite_quality["intent_preserved"]:
                started = perf_counter()
                retry_results = self.store.search(
                    rewritten_query,
                    self.embedder,
                    self.top_k * 2,
                    self.reranker,
                )
                retry_filtered = await self._filter_relevant(question, retry_results)
                rewritten_evidence_score = _evidence_score(retry_filtered or retry_results)
                evidence_improved = (
                    rewritten_evidence_score
                    > initial_evidence_score + self.rewrite_evidence_margin
                )
                second_retrieval_ms = _elapsed_ms(started)
                steps.append(
                    PipelineStep(
                        name="Second Retrieval",
                        status="accepted" if evidence_improved else "rejected",
                        description=(
                            "Searched again with the rewrite and accepted it only if evidence quality improved."
                        ),
                        duration_ms=second_retrieval_ms,
                        details={
                            "matches": len(retry_results),
                            "kept_chunks": len(retry_filtered),
                            "best_score": _best_score(retry_results),
                            "initial_evidence_score": initial_evidence_score,
                            "rewritten_evidence_score": rewritten_evidence_score,
                            "evidence_margin": self.rewrite_evidence_margin,
                            "evidence_improved": evidence_improved,
                        },
                    )
                )
                if evidence_improved:
                    filtered = retry_filtered
                    correction_applied = True
                else:
                    steps.append(
                        PipelineStep(
                            name="Rewrite Decision",
                            status="rejected",
                            description=(
                                "The rewrite preserved intent but did not improve retrieval evidence enough."
                            ),
                            details={"fallback": "initial retrieval"},
                        )
                    )
            else:
                steps.append(
                    PipelineStep(
                        name="Rewrite Decision",
                        status="rejected",
                        description="The rewrite was rejected because it drifted from the original query intent.",
                        details={"fallback": "initial retrieval"},
                    )
                )
        context_results = filtered[: self.top_k] if filtered else initial_results[: self.top_k]
        started = perf_counter()
        prompt = _build_answer_prompt(question, context_results, personalization)
        build_ms = _elapsed_ms(started)

        started = perf_counter()
        response = await self.llm.generate(prompt)
        generate_ms = _elapsed_ms(started)

        steps.extend(
            [
                PipelineStep(
                    name="Build Corrected Context",
                    status="completed",
                    description="Selected the final chunks that will be shown to the LLM.",
                    duration_ms=build_ms,
                    details={
                        "sources": _source_labels(context_results),
                        **_personalization_details(personalization),
                    },
                ),
                PipelineStep(
                    name="Generate Answer",
                    status="completed",
                    description="Sent the question and corrected context to the configured LLM interface.",
                    duration_ms=generate_ms,
                    details={"context_chunks": len(context_results)},
                ),
            ]
        )
        answer = Answer(
            question=question,
            answer=response,
            sources=context_results,
            pipeline="corrective",
            steps=steps,
            personalization=personalization,
            rewritten_query=rewritten_query,
            correction_applied=correction_applied,
        )
        return await self._run_judge(answer)

    async def _rewrite_query(self, question: str) -> str:
        rewritten = await self.llm.generate(REWRITE_PROMPT.format(question=question))
        return rewritten.splitlines()[0].strip().strip('"') or question

    def _validate_rewrite(self, question: str, rewritten_query: str) -> dict[str, float | bool]:
        intent_similarity = _query_similarity(self.embedder, question, rewritten_query)
        term_preservation = _term_preservation(question, rewritten_query)
        polarity_changed = _query_polarity(question) != _query_polarity(rewritten_query)
        intent_preserved = (
            bool(rewritten_query.strip())
            and not polarity_changed
            and (
                intent_similarity >= self.rewrite_intent_threshold
                or term_preservation >= 0.8
            )
        )
        return {
            "intent_similarity": round(intent_similarity, 4),
            "term_preservation": round(term_preservation, 4),
            "polarity_changed": polarity_changed,
            "intent_preserved": intent_preserved,
        }

    def _is_weak_evidence(self, results: list[SearchResult], evidence_score: float) -> bool:
        return bool(self._evidence_trigger_reasons(results, evidence_score))

    def _evidence_trigger_reasons(
        self,
        results: list[SearchResult],
        evidence_score: float,
    ) -> list[str]:
        reasons: list[str] = []
        if not results:
            reasons.append("no relevant chunks survived grading")
        if _mean_score(results) < self.relevance_threshold:
            reasons.append("mean vector relevance is below threshold")
        if self.reranker is not None and evidence_score < self.reranker_evidence_threshold:
            reasons.append("reranker evidence is below threshold")
        return reasons

    async def _filter_relevant(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        kept: list[SearchResult] = []
        for result in results:
            if _relevance_score(result) >= self.relevance_threshold:
                kept.append(result)
                continue
            prompt = GRADE_PROMPT.format(question=question, context=result.chunk.text[:1200])
            grade = (await self.llm.generate(prompt)).lower()
            if "relevant" in grade and "irrelevant" not in grade:
                kept.append(result)
        return kept


class PlannerRAGPipeline(StandardRAGPipeline):
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        top_k: int = 5,
        max_sub_questions: int = 4,
        reranker: Reranker | None = None,
        judge: Judge | None = None,
    ) -> None:
        super().__init__(
            store=store,
            embedder=embedder,
            llm=llm,
            top_k=top_k,
            reranker=reranker,
            judge=judge,
        )
        self.max_sub_questions = max_sub_questions

    async def answer(
        self,
        question: str,
        personalization: PersonalizationContext | None = None,
    ) -> Answer:
        started = perf_counter()
        sub_questions = await self._decompose_question(question)
        plan_ms = _elapsed_ms(started)

        steps = [
            PipelineStep(
                name="Decompose Query",
                status="completed",
                description="Split the original question into focused retrieval tasks.",
                duration_ms=plan_ms,
                details={
                    "sub_questions": sub_questions,
                    "sub_question_count": len(sub_questions),
                    "max_sub_questions": self.max_sub_questions,
                },
            )
        ]

        retrieval_lists: list[list[SearchResult]] = []
        started = perf_counter()
        for index, sub_question in enumerate(sub_questions, start=1):
            sub_results = self.store.search(sub_question, self.embedder, self.top_k, self.reranker)
            retrieval_lists.append(sub_results)
            steps.append(
                PipelineStep(
                    name=f"Retrieve Sub-query {index}",
                    status="completed",
                    description="Retrieved evidence for one decomposed search task.",
                    details={
                        "sub_question": sub_question,
                        "matches": len(sub_results),
                        "best_score": _best_score(sub_results),
                        "sources": _source_labels(sub_results),
                    },
                )
            )
        retrieve_ms = _elapsed_ms(started)

        started = perf_counter()
        fused_results = _fuse_planner_results(retrieval_lists, self.top_k * 3)
        if self.reranker is not None:
            fused_results = self.reranker.rerank(question, fused_results, self.top_k)
        else:
            fused_results = fused_results[: self.top_k]
        fuse_ms = _elapsed_ms(started)
        steps.append(
            PipelineStep(
                name="Fuse Sub-query Evidence",
                status="completed",
                description="Merged evidence from every sub-query into one ranked context set.",
                duration_ms=fuse_ms,
                details={
                    "retrieval_duration_ms": retrieve_ms,
                    "candidate_lists": len(retrieval_lists),
                    "final_sources": _source_labels(fused_results),
                    "reranker": self.reranker is not None,
                },
            )
        )

        started = perf_counter()
        prompt = _build_answer_prompt(question, fused_results, personalization)
        build_ms = _elapsed_ms(started)

        started = perf_counter()
        response = await self.llm.generate(prompt)
        generate_ms = _elapsed_ms(started)

        steps.extend(
            [
                PipelineStep(
                    name="Build Planned Context",
                    status="completed",
                    description="Packed the fused sub-query evidence into the answer prompt.",
                    duration_ms=build_ms,
                    details={
                        "context_chunks": len(fused_results),
                        **_personalization_details(personalization),
                    },
                ),
                PipelineStep(
                    name="Generate Answer",
                    status="completed",
                    description="Synthesized a single answer from the planned retrieval context.",
                    duration_ms=generate_ms,
                    details={"sub_questions": sub_questions},
                ),
            ]
        )

        answer = Answer(
            question=question,
            answer=response,
            sources=fused_results,
            pipeline="planner",
            steps=steps,
            personalization=personalization,
        )
        return await self._run_judge(answer)

    async def _decompose_question(self, question: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            max_sub_questions=self.max_sub_questions,
        )
        response = await self.llm.generate(prompt)
        parsed = _parse_sub_questions(response, self.max_sub_questions)
        return parsed or [question]


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "No retrieved context."
    return "\n\n".join(
        f"[{result.chunk.source}: chunk {result.chunk.metadata.get('chunk', 0)} | "
        f"score {_relevance_score(result):.3f} | rrf {result.raw_scores.get('rrf', 0.0):.4f} | "
        f"reranker {result.raw_scores.get('reranker', 0.0):.3f}]\n{result.chunk.text}"
        for result in results
    )


def _build_answer_prompt(
    question: str,
    results: list[SearchResult],
    personalization: PersonalizationContext | None,
) -> str:
    return ANSWER_PROMPT.format(
        question=question,
        context=_format_context(results),
        personalization=format_personalization_context(personalization),
    )


def _personalization_details(
    personalization: PersonalizationContext | None,
) -> dict[str, str | bool | list[str]]:
    if personalization is None:
        return {
            "memory_mode": "stateless",
            "memory_loaded": False,
            "memory_saved": False,
            "personalization_preferences": [],
        }
    return {
        "memory_mode": personalization.mode,
        "memory_loaded": personalization.memory_loaded,
        "memory_saved": personalization.memory_saved,
        "personalization_preferences": summarize_preferences(personalization),
    }


def _mean_score(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return sum(_relevance_score(result) for result in results) / len(results)


def _best_score(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return max(_relevance_score(result) for result in results)


def _source_labels(results: list[SearchResult]) -> list[str]:
    return [
        f"{result.chunk.source} chunk {result.chunk.metadata.get('chunk', 0)} "
        f"via {'+'.join(result.retrieval_sources) or 'unknown'}"
        for result in results
    ]


def _fuse_planner_results(
    result_lists: list[list[SearchResult]],
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
            hits[chunk_id] = hits.get(chunk_id, 0) + 1
            if chunk_id not in fused:
                fused[chunk_id] = result.model_copy(deep=True)
            else:
                existing = fused[chunk_id]
                existing.retrieval_sources = sorted(
                    set(existing.retrieval_sources + result.retrieval_sources)
                )
                existing.raw_scores.update(result.raw_scores)
                existing.ranks.update(result.ranks)

    ranked = sorted(fused.values(), key=lambda result: scores[result.chunk.id], reverse=True)
    for rank, result in enumerate(ranked, start=1):
        score = scores[result.chunk.id]
        result.score = score
        result.raw_scores["planner_rrf"] = score
        result.raw_scores["subquery_hits"] = float(hits[result.chunk.id])
        result.ranks["planner"] = rank
    return ranked[:top_k]


def _parse_sub_questions(response: str, max_items: int) -> list[str]:
    sub_questions: list[str] = []
    for line in response.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        cleaned = cleaned.strip('"')
        if not cleaned:
            continue
        if cleaned.lower().startswith(("sub-questions:", "sub questions:")):
            continue
        if cleaned not in sub_questions:
            sub_questions.append(cleaned)
        if len(sub_questions) >= max_items:
            break
    if not sub_questions and response.strip():
        sub_questions.append(response.strip().splitlines()[0].strip().strip('"'))
    return sub_questions[:max_items]


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def _relevance_score(result: SearchResult) -> float:
    return result.raw_scores.get("vector", result.score)


def _evidence_score(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    reranker_scores = [result.raw_scores["reranker"] for result in results if "reranker" in result.raw_scores]
    if reranker_scores:
        return max(reranker_scores)
    return _mean_score(results)


def _query_similarity(embedder: Embedder, original: str, rewritten: str) -> float:
    vectors = embedder.embed([original, rewritten])
    return float(np.dot(vectors[0], vectors[1]))


def _query_polarity(query: str) -> str:
    lowered = f" {query.lower()} "
    negative_markers = (" not ", " unsupported ", "not supported", "doesn't", "do not", "cannot")
    return "negative" if any(marker in lowered for marker in negative_markers) else "positive"


def _term_preservation(original: str, rewritten: str) -> float:
    original_terms = _important_terms(original)
    if not original_terms:
        return 1.0
    rewritten_terms = _important_terms(rewritten)
    return len(original_terms.intersection(rewritten_terms)) / len(original_terms)


def _important_terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "are",
        "can",
        "could",
        "does",
        "for",
        "from",
        "how",
        "into",
        "is",
        "me",
        "please",
        "tell",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]*", text.lower())
        if token not in stopwords and len(token) > 2
    }
