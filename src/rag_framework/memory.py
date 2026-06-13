import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from rag_framework.models import PersonalizationContext

MemoryMode = Literal["stateless", "stateful"]

MAX_KEY_LENGTH = 40
MAX_VALUE_LENGTH = 220
MAX_TEXT_LENGTH = 800
MAX_LIST_ITEMS = 12
MAX_ITEM_LENGTH = 220


class MemoryManager:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path

    def build_context(
        self,
        *,
        mode: MemoryMode,
        user_id: str | None = None,
        session_preferences: dict[str, str] | None = None,
        remember_preferences: bool = False,
        conversation_memory: dict[str, object] | None = None,
        remember_conversation_memory: bool = False,
        scratchpad_memory: dict[str, object] | None = None,
        remember_scratchpad_memory: bool = False,
    ) -> PersonalizationContext:
        session = _sanitize_preferences(session_preferences or {})
        conversation_update = _sanitize_conversation_memory(conversation_memory or {})
        scratchpad_update = _sanitize_scratchpad_memory(scratchpad_memory or {})
        if mode == "stateless":
            return PersonalizationContext(
                mode=mode,
                user_id=None,
                preferences=session,
                session_preferences=session,
                **conversation_update,
                **scratchpad_update,
                memory_loaded=False,
                memory_saved=False,
            )

        clean_user_id = _sanitize_user_id(user_id)
        if not clean_user_id:
            raise ValueError("Stateful memory requires a user_id.")

        store = self._read_store()
        user_memory = store.get(clean_user_id, {})
        persisted = _sanitize_preferences(user_memory.get("preferences", {}))
        persisted_conversation = _sanitize_conversation_memory(
            user_memory.get("conversation", {})
        )
        persisted_scratchpad = _sanitize_scratchpad_memory(user_memory.get("scratchpad", {}))
        preferences = {**persisted, **session}
        conversation = _merge_conversation_memory(persisted_conversation, conversation_update)
        scratchpad = _merge_scratchpad_memory(persisted_scratchpad, scratchpad_update)
        memory_saved = False
        conversation_saved = False
        scratchpad_saved = False

        if remember_preferences and session:
            memory_saved = True
        if remember_conversation_memory and _has_conversation_memory(conversation_update):
            conversation_saved = True
        if remember_scratchpad_memory and _has_scratchpad_memory(scratchpad_update):
            scratchpad_saved = True

        if memory_saved or conversation_saved or scratchpad_saved:
            next_memory = dict(user_memory)
            if memory_saved:
                next_memory["preferences"] = preferences
            if conversation_saved:
                next_memory["conversation"] = conversation
            if scratchpad_saved:
                next_memory["scratchpad"] = scratchpad
            next_memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            store[clean_user_id] = next_memory
            self._write_store(store)
            memory_saved = True

        return PersonalizationContext(
            mode=mode,
            user_id=clean_user_id,
            preferences=preferences,
            persisted_preferences=persisted,
            session_preferences=session,
            **conversation,
            **scratchpad,
            memory_loaded=bool(persisted),
            memory_saved=memory_saved,
            conversation_memory_loaded=_has_conversation_memory(persisted_conversation),
            conversation_memory_saved=conversation_saved,
            scratchpad_memory_loaded=_has_scratchpad_memory(persisted_scratchpad),
            scratchpad_memory_saved=scratchpad_saved,
        )

    def _read_store(self) -> dict[str, dict[str, object]]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(user_id): memory
            for user_id, memory in payload.items()
            if isinstance(memory, dict)
        }

    def _write_store(self, store: dict[str, dict[str, object]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.store_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(store, indent=2, sort_keys=True))
        temporary_path.replace(self.store_path)


def format_personalization_context(context: PersonalizationContext | None) -> str:
    if context is None:
        return (
            "No personalization preferences were provided. Use a neutral, concise style. "
            "Do not treat user memory as factual source evidence."
        )

    lines = [
        f"Memory mode: {context.mode}.",
        "Use these preferences only to adapt tone, depth, formatting, and examples.",
        "Do not treat user memory as retrieved factual evidence or cite it as a source.",
    ]
    if context.user_id:
        lines.append(f"User ID: {context.user_id}.")
    if context.preferences:
        lines.append("Style preferences:")
        for key, value in sorted(context.preferences.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("No style preferences were provided.")
    if _has_conversation_context(context):
        lines.append("\nConversation memory:")
        if context.conversation_summary:
            lines.append(f"- summary: {context.conversation_summary}")
        if context.current_goal:
            lines.append(f"- current goal: {context.current_goal}")
        if context.user_intent:
            lines.append(f"- user intent: {context.user_intent}")
        if context.recent_topics:
            lines.append(f"- recent topics: {', '.join(context.recent_topics)}")
    if _has_scratchpad_context(context):
        lines.append("\nScratchpad memory:")
        for fact in context.scratchpad_facts:
            lines.append(f"- fact: {fact}")
        for decision in context.scratchpad_decisions:
            lines.append(f"- decision: {decision}")
        for question in context.scratchpad_open_questions:
            lines.append(f"- open question: {question}")
    return "\n".join(lines)


def summarize_preferences(context: PersonalizationContext | None) -> list[str]:
    if context is None:
        return []
    return [f"{key}: {value}" for key, value in sorted(context.preferences.items())]


def summarize_scratchpad(context: PersonalizationContext | None) -> list[str]:
    if context is None:
        return []
    items = [f"fact: {item}" for item in context.scratchpad_facts]
    items.extend(f"decision: {item}" for item in context.scratchpad_decisions)
    items.extend(f"open question: {item}" for item in context.scratchpad_open_questions)
    return items


def _sanitize_preferences(preferences: dict[str, object]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in preferences.items():
        clean_key = _sanitize_key(str(key))
        clean_value = " ".join(str(value).split())[:MAX_VALUE_LENGTH]
        if clean_key and clean_value:
            clean[clean_key] = clean_value
    return clean


def _sanitize_conversation_memory(memory: object) -> dict[str, object]:
    if not isinstance(memory, dict):
        return {
            "conversation_summary": None,
            "current_goal": None,
            "user_intent": None,
            "recent_topics": [],
        }
    return {
        "conversation_summary": _sanitize_optional_text(memory.get("conversation_summary")),
        "current_goal": _sanitize_optional_text(memory.get("current_goal")),
        "user_intent": _sanitize_optional_text(memory.get("user_intent")),
        "recent_topics": _sanitize_list(memory.get("recent_topics")),
    }


def _sanitize_scratchpad_memory(memory: object) -> dict[str, list[str]]:
    if not isinstance(memory, dict):
        return {
            "scratchpad_facts": [],
            "scratchpad_decisions": [],
            "scratchpad_open_questions": [],
        }
    return {
        "scratchpad_facts": _sanitize_list(memory.get("facts") or memory.get("scratchpad_facts")),
        "scratchpad_decisions": _sanitize_list(
            memory.get("decisions") or memory.get("scratchpad_decisions")
        ),
        "scratchpad_open_questions": _sanitize_list(
            memory.get("open_questions") or memory.get("scratchpad_open_questions")
        ),
    }


def _merge_conversation_memory(
    persisted: dict[str, object],
    update: dict[str, object],
) -> dict[str, object]:
    return {
        "conversation_summary": update.get("conversation_summary")
        or persisted.get("conversation_summary"),
        "current_goal": update.get("current_goal") or persisted.get("current_goal"),
        "user_intent": update.get("user_intent") or persisted.get("user_intent"),
        "recent_topics": _merge_unique(
            persisted.get("recent_topics", []),
            update.get("recent_topics", []),
        ),
    }


def _merge_scratchpad_memory(
    persisted: dict[str, list[str]],
    update: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        "scratchpad_facts": _merge_unique(
            persisted.get("scratchpad_facts", []),
            update.get("scratchpad_facts", []),
        ),
        "scratchpad_decisions": _merge_unique(
            persisted.get("scratchpad_decisions", []),
            update.get("scratchpad_decisions", []),
        ),
        "scratchpad_open_questions": _merge_unique(
            persisted.get("scratchpad_open_questions", []),
            update.get("scratchpad_open_questions", []),
        ),
    }


def _sanitize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())[:MAX_TEXT_LENGTH]
    return cleaned or None


def _sanitize_list(value: object) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else str(value).splitlines()
    clean: list[str] = []
    for item in values:
        cleaned = " ".join(str(item).split())[:MAX_ITEM_LENGTH]
        if cleaned and cleaned not in clean:
            clean.append(cleaned)
        if len(clean) >= MAX_LIST_ITEMS:
            break
    return clean


def _merge_unique(existing: object, incoming: object) -> list[str]:
    merged: list[str] = []
    for item in [*_sanitize_list(existing), *_sanitize_list(incoming)]:
        if item not in merged:
            merged.append(item)
        if len(merged) >= MAX_LIST_ITEMS:
            break
    return merged


def _has_conversation_memory(memory: dict[str, object]) -> bool:
    return bool(
        memory.get("conversation_summary")
        or memory.get("current_goal")
        or memory.get("user_intent")
        or memory.get("recent_topics")
    )


def _has_scratchpad_memory(memory: dict[str, list[str]]) -> bool:
    return bool(
        memory.get("scratchpad_facts")
        or memory.get("scratchpad_decisions")
        or memory.get("scratchpad_open_questions")
    )


def _has_conversation_context(context: PersonalizationContext) -> bool:
    return bool(
        context.conversation_summary
        or context.current_goal
        or context.user_intent
        or context.recent_topics
    )


def _has_scratchpad_context(context: PersonalizationContext) -> bool:
    return bool(
        context.scratchpad_facts
        or context.scratchpad_decisions
        or context.scratchpad_open_questions
    )


def _sanitize_key(key: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", key.strip().lower())
    return normalized.strip("_")[:MAX_KEY_LENGTH]


def _sanitize_user_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9_.@-]+", "-", user_id.strip())
    return normalized[:80] or None
