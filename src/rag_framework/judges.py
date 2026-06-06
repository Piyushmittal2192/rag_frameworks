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
        payload["unsupported_claims"] = _coerce_string_list(
            payload.get("unsupported_claims", [])
        )
        payload["citation_issues"] = _coerce_string_list(payload.get("citation_issues", []))
        payload["reason"] = str(payload.get("reason") or "No judge explanation returned.")
        _enforce_consistent_payload(payload)
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
    if isinstance(value, str) and value.strip().endswith("%"):
        value = value.strip().removesuffix("%")
        try:
            return _clamp_score(float(value) / 100)
        except ValueError:
            return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _normalize_verdict(value: object) -> str:
    verdict = str(value).strip().lower().replace(" ", "_").strip(".:")
    valid = {"grounded", "partially_grounded", "unsupported", "judge_error"}
    return verdict if verdict in valid else "unsupported"


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"none", "n/a", "na", "no", "[]"}:
            return []
        return [cleaned]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                text = item.get("claim") or item.get("issue") or item.get("reason") or item
            else:
                text = item
            cleaned = str(text).strip()
            if cleaned and cleaned.lower() not in {"none", "n/a", "na"}:
                items.append(cleaned)
        return items
    return [str(value)]


def _enforce_consistent_payload(payload: dict) -> None:
    has_issues = bool(payload["unsupported_claims"] or payload["citation_issues"])
    if payload["verdict"] == "grounded" and has_issues:
        payload["verdict"] = "partially_grounded"
        payload["faithfulness_score"] = min(payload["faithfulness_score"], 0.75)
    if payload["verdict"] == "unsupported":
        payload["faithfulness_score"] = min(payload["faithfulness_score"], 0.4)
