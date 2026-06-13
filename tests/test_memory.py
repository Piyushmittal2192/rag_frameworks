from pathlib import Path

import pytest

from rag_framework.embeddings import TfidfEmbedder
from rag_framework.memory import MemoryManager, format_personalization_context
from rag_framework.models import DocumentChunk, PersonalizationContext
from rag_framework.pipelines import StandardRAGPipeline
from rag_framework.vector_store import VectorStore


class CaptureLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "captured answer"


def test_stateless_memory_uses_request_preferences_only(tmp_path: Path):
    manager = MemoryManager(tmp_path / "users.json")

    context = manager.build_context(
        mode="stateless",
        session_preferences={"Depth": " Concise technical ", "unsafe key!": "ok"},
        remember_preferences=True,
    )

    assert context.mode == "stateless"
    assert context.user_id is None
    assert context.preferences == {"depth": "Concise technical", "unsafe_key": "ok"}
    assert context.memory_loaded is False
    assert context.memory_saved is False
    assert not (tmp_path / "users.json").exists()


def test_stateful_memory_persists_and_reloads_preferences(tmp_path: Path):
    manager = MemoryManager(tmp_path / "users.json")

    first = manager.build_context(
        mode="stateful",
        user_id="piyush@example.com",
        session_preferences={"format": "bullets"},
        remember_preferences=True,
    )
    second = manager.build_context(
        mode="stateful",
        user_id="piyush@example.com",
        session_preferences={"depth": "detailed"},
    )

    assert first.memory_saved is True
    assert second.memory_loaded is True
    assert second.preferences == {"format": "bullets", "depth": "detailed"}


def test_stateful_conversation_memory_persists_per_user(tmp_path: Path):
    manager = MemoryManager(tmp_path / "users.json")

    first = manager.build_context(
        mode="stateful",
        user_id="piyush@example.com",
        conversation_memory={
            "conversation_summary": "Built RAG app and added memory design.",
            "current_goal": "Add per-user scratchpad memory.",
            "user_intent": "Create an educational deployable RAG framework.",
            "recent_topics": ["RAG", "memory", "faithfulness"],
        },
        remember_conversation_memory=True,
    )
    second = manager.build_context(mode="stateful", user_id="piyush@example.com")

    assert first.conversation_memory_saved is True
    assert second.conversation_memory_loaded is True
    assert second.conversation_summary == "Built RAG app and added memory design."
    assert second.current_goal == "Add per-user scratchpad memory."
    assert second.user_intent == "Create an educational deployable RAG framework."
    assert second.recent_topics == ["RAG", "memory", "faithfulness"]


def test_stateful_scratchpad_memory_merges_facts_decisions_and_questions(tmp_path: Path):
    manager = MemoryManager(tmp_path / "users.json")

    first = manager.build_context(
        mode="stateful",
        user_id="piyush@example.com",
        scratchpad_memory={
            "facts": ["Repo is Piyushmittal2192/rag_frameworks"],
            "decisions": ["Memory is not cited as retrieval evidence"],
            "open_questions": ["Should memory influence retrieval expansion?"],
        },
        remember_scratchpad_memory=True,
    )
    second = manager.build_context(
        mode="stateful",
        user_id="piyush@example.com",
        scratchpad_memory={"facts": ["Local app can run on port 8010"]},
    )

    assert first.scratchpad_memory_saved is True
    assert second.scratchpad_memory_loaded is True
    assert second.scratchpad_facts == [
        "Repo is Piyushmittal2192/rag_frameworks",
        "Local app can run on port 8010",
    ]
    assert second.scratchpad_decisions == ["Memory is not cited as retrieval evidence"]
    assert second.scratchpad_open_questions == ["Should memory influence retrieval expansion?"]


def test_stateful_memory_requires_user_id(tmp_path: Path):
    manager = MemoryManager(tmp_path / "users.json")

    with pytest.raises(ValueError, match="user_id"):
        manager.build_context(mode="stateful")


@pytest.mark.asyncio
async def test_pipeline_adds_personalization_to_answer_prompt():
    chunks = [DocumentChunk(id="1", text="RAG uses retrieved context.", source="rag.txt")]
    embedder = TfidfEmbedder()
    store = VectorStore.build(chunks, embedder)
    llm = CaptureLLM()
    pipeline = StandardRAGPipeline(store, embedder, llm, top_k=1)
    personalization = PersonalizationContext(
        mode="stateless",
        preferences={"depth": "technical"},
        session_preferences={"depth": "technical"},
    )

    answer = await pipeline.answer("What does RAG use?", personalization=personalization)

    assert answer.personalization == personalization
    assert "Personalization:" in llm.prompt
    assert "- depth: technical" in llm.prompt
    assert "Do not treat user memory as retrieved factual evidence" in llm.prompt
    build_step = next(step for step in answer.steps if step.name == "Build Context")
    assert build_step.details["memory_mode"] == "stateless"


def test_personalization_prompt_has_neutral_default():
    prompt_block = format_personalization_context(None)

    assert "No personalization preferences" in prompt_block
    assert "Do not treat user memory as factual source evidence" in prompt_block


def test_personalization_prompt_includes_conversation_and_scratchpad_context():
    context = PersonalizationContext(
        mode="stateful",
        user_id="local-demo",
        conversation_summary="We added a RAG UI and trust layer.",
        current_goal="Add working memory.",
        user_intent="Build a transparent learning app.",
        recent_topics=["planner rag", "memory"],
        scratchpad_facts=["GitHub repo is Piyushmittal2192/rag_frameworks"],
        scratchpad_decisions=["Do not cite memory as evidence"],
        scratchpad_open_questions=["Should memory influence retrieval?"],
    )

    prompt_block = format_personalization_context(context)

    assert "Conversation memory:" in prompt_block
    assert "current goal: Add working memory." in prompt_block
    assert "Scratchpad memory:" in prompt_block
    assert "decision: Do not cite memory as evidence" in prompt_block
