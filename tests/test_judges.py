import pytest

from rag_framework.judges import LLMFaithfulnessJudge, _parse_judge_response
from rag_framework.models import DocumentChunk, SearchResult


class JsonJudgeLLM:
    async def generate(self, prompt: str) -> str:
        return """
        {
          "verdict": "grounded",
          "faithfulness_score": 0.91,
          "unsupported_claims": [],
          "citation_issues": [],
          "reason": "All claims are supported."
        }
        """


class BrokenJudgeLLM:
    async def generate(self, prompt: str) -> str:
        return "not json"


def test_parse_judge_response_clamps_score_and_normalizes_verdict():
    result = _parse_judge_response(
        '{"verdict":"mystery","faithfulness_score":2.5,'
        '"unsupported_claims":[],"citation_issues":[],"reason":"ok"}'
    )

    assert result.verdict == "unsupported"
    assert result.faithfulness_score == 0.4


def test_parse_judge_response_handles_malformed_output():
    result = _parse_judge_response("not json")

    assert result.verdict == "judge_error"
    assert result.faithfulness_score == 0.0


def test_parse_judge_response_tolerates_common_llm_schema_variants():
    result = _parse_judge_response(
        """
        ```json
        {
          "verdict": "Grounded.",
          "faithfulness_score": "92%",
          "unsupported_claims": "None",
          "citation_issues": [{"issue": "None"}],
          "reason": null
        }
        ```
        """
    )

    assert result.verdict == "grounded"
    assert result.faithfulness_score == 0.92
    assert result.unsupported_claims == []
    assert result.citation_issues == []
    assert result.reason == "No judge explanation returned."


def test_parse_judge_response_downgrades_inconsistent_grounded_verdict():
    result = _parse_judge_response(
        """
        {
          "verdict": "grounded",
          "faithfulness_score": 1.0,
          "unsupported_claims": [],
          "citation_issues": ["The citation does not support one claim."],
          "reason": "One citation is weak."
        }
        """
    )

    assert result.verdict == "partially_grounded"
    assert result.faithfulness_score == 0.75


@pytest.mark.asyncio
async def test_llm_faithfulness_judge_returns_result():
    judge = LLMFaithfulnessJudge(JsonJudgeLLM())
    sources = [
        SearchResult(
            chunk=DocumentChunk(id="1", text="RAG uses retrieved context.", source="rag.txt"),
            score=1.0,
        )
    ]

    result = await judge.evaluate("What does RAG use?", "RAG uses retrieved context.", sources)

    assert result.verdict == "grounded"
    assert result.faithfulness_score == 0.91


@pytest.mark.asyncio
async def test_llm_faithfulness_judge_falls_back_on_broken_output():
    judge = LLMFaithfulnessJudge(BrokenJudgeLLM())

    result = await judge.evaluate("Question?", "Answer.", [])

    assert result.verdict == "judge_error"
