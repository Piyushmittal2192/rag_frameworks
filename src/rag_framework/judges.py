import json
import re
from typing import Protocol

from pydantic import ValidationError

from rag_framework.llms import LLM
from rag_framework.models import JudgeResult, SearchResult
from rag_framework.prompts import JUDGE_PROMPT


class Judge(Protocol):
    async def evaluate(
        self,
        question: str,
        answer: str,
        sources: list[SearchResult],
    ) -> JudgeResult:
        ...


class LLMFaithfulnessJudge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def evaluate(
        self,
        question: str,
        answer: str,
        sources: list[SearchResult],
    ) -> JudgeResult:
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            context=_format_judge_context(sources),
        )
        response = await self.llm.generate(prompt)
        return _parse_judge_response(response)


def _format_judge_context(sources: list[SearchResult]) -> str:
    if not sources:
        return "No retrieved context."
    return "\n\n".join(
        f"[{source.chunk.source}: chunk {source.chunk.metadata.get('chunk', 0)}]\n"
        f"{source.chunk.text}"
        for source in sources
    )


def _parse_judge_response(response: str) -> JudgeResult:
    try:
        payload = json.loads(_extract_json(response))
        payload["faithfulness_score"] = _clamp_score(payload.get("faithfulness_score", 0.0))
        payload["verdict"] = _normalize_verdict(payload.get("verdict", "unsupported"))
        return JudgeResult.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError):
        return JudgeResult(
            verdict="judge_error",
            faithfulness_score=0.0,
            unsupported_claims=[],
            citation_issues=[],
            reason="Judge response could not be parsed as the expected JSON schema.",
        )


def _extract_json(response: str) -> str:
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response.")
    return match.group(0)


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _normalize_verdict(value: object) -> str:
    verdict = str(value).strip().lower()
    valid = {"grounded", "partially_grounded", "unsupported", "judge_error"}
    return verdict if verdict in valid else "unsupported"
