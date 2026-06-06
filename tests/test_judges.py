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
    assert result.faithfulness_score == 1.0


def test_parse_judge_response_handles_malformed_output():
    result = _parse_judge_response("not json")

    assert result.verdict == "judge_error"
    assert result.faithfulness_score == 0.0


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
