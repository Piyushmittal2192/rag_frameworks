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
DEFAULT_CONFIDENCE = 0.75
REINFORCED_CONFIDENCE_STEP = 0.05
REINFORCED_IMPORTANCE_STEP = 0.03
MAX_REINFORCEMENT_BOOST = 0.3

MEMORY_DEFAULTS = {
    "fact": {"importance": 0.65, "decay_rate": 0.01},
    "decision": {"importance": 0.9, "decay_rate": 0.001},
    "open_question": {"importance": 0.75, "decay_rate": 0.02},
}


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
        persisted_scratchpad_items = _sanitize_scratchpad_items(user_memory.get("scratchpad", {}))
        persisted_scratchpad = _select_scratchpad_context(persisted_scratchpad_items)
        preferences = {**persisted, **session}
        conversation = _merge_conversation_memory(persisted_conversation, conversation_update)
        scratchpad_items = _merge_scratchpad_items(persisted_scratchpad_items, scratchpad_update)
        scratchpad = _select_scratchpad_context(scratchpad_items)
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
                next_memory["scratchpad"] = _serialize_scratchpad_items(scratchpad_items)
            next_memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            store[clean_user_id] = next_memory
            self._write_store(store)
            memory_saved = True
        elif _has_scratchpad_memory(persisted_scratchpad):
            # Reinforce memories that were useful enough to enter the prompt context.
            next_memory = dict(user_memory)
            next_memory["scratchpad"] = _serialize_scratchpad_items(
                _reinforce_selected_items(scratchpad_items, scratchpad)
            )
            next_memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            store[clean_user_id] = next_memory
            self._write_store(store)

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


def _sanitize_scratchpad_items(memory: object) -> dict[str, list[dict[str, object]]]:
    scratchpad = _scratchpad_payload(memory)
    return {
        "fact": _sanitize_memory_item_list(
            scratchpad.get("facts") or scratchpad.get("scratchpad_facts"),
            "fact",
        ),
        "decision": _sanitize_memory_item_list(
            scratchpad.get("decisions") or scratchpad.get("scratchpad_decisions"),
            "decision",
        ),
        "open_question": _sanitize_memory_item_list(
            scratchpad.get("open_questions") or scratchpad.get("scratchpad_open_questions"),
            "open_question",
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


def _merge_scratchpad_items(
    persisted: dict[str, list[dict[str, object]]],
    update: dict[str, list[str]],
) -> dict[str, list[dict[str, object]]]:
    merged: dict[str, list[dict[str, object]]] = {
        "fact": list(persisted.get("fact", [])),
        "decision": list(persisted.get("decision", [])),
        "open_question": list(persisted.get("open_question", [])),
    }
    for memory_type, update_key in (
        ("fact", "scratchpad_facts"),
        ("decision", "scratchpad_decisions"),
        ("open_question", "scratchpad_open_questions"),
    ):
        existing_by_text = {
            str(item["text"]).casefold(): item
            for item in merged[memory_type]
            if item.get("text")
        }
        for text in update.get(update_key, []):
            key = text.casefold()
            if key in existing_by_text:
                _reinforce_item(existing_by_text[key])
            else:
                item = _new_memory_item(text, memory_type)
                merged[memory_type].append(item)
                existing_by_text[key] = item
        merged[memory_type] = _rank_memory_items(merged[memory_type])[:MAX_LIST_ITEMS]
    return merged


def _select_scratchpad_context(
    items: dict[str, list[dict[str, object]]],
) -> dict[str, list[str]]:
    return {
        "scratchpad_facts": [
            str(item["text"])
            for item in _rank_memory_items(items.get("fact", []))[:MAX_LIST_ITEMS]
        ],
        "scratchpad_decisions": [
            str(item["text"])
            for item in _rank_memory_items(items.get("decision", []))[:MAX_LIST_ITEMS]
        ],
        "scratchpad_open_questions": [
            str(item["text"])
            for item in _rank_memory_items(items.get("open_question", []))[:MAX_LIST_ITEMS]
        ],
    }


def _serialize_scratchpad_items(
    items: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    return {
        "scratchpad_facts": _rank_memory_items(items.get("fact", []))[:MAX_LIST_ITEMS],
        "scratchpad_decisions": _rank_memory_items(items.get("decision", []))[:MAX_LIST_ITEMS],
        "scratchpad_open_questions": _rank_memory_items(
            items.get("open_question", [])
        )[:MAX_LIST_ITEMS],
    }


def _reinforce_selected_items(
    items: dict[str, list[dict[str, object]]],
    selected: dict[str, list[str]],
) -> dict[str, list[dict[str, object]]]:
    selected_by_type = {
        "fact": {item.casefold() for item in selected.get("scratchpad_facts", [])},
        "decision": {item.casefold() for item in selected.get("scratchpad_decisions", [])},
        "open_question": {
            item.casefold() for item in selected.get("scratchpad_open_questions", [])
        },
    }
    for memory_type, type_items in items.items():
        for item in type_items:
            if str(item.get("text", "")).casefold() in selected_by_type.get(memory_type, set()):
                _mark_item_used(item)
    return items


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


def _scratchpad_payload(memory: object) -> dict[str, object]:
    if not isinstance(memory, dict):
        return {}
    # Support both the original flat scratchpad shape and the new structured store shape.
    if "scratchpad_facts" in memory or "scratchpad_decisions" in memory:
        return memory
    return memory


def _sanitize_memory_item_list(value: object, memory_type: str) -> list[dict[str, object]]:
    if value is None:
        return []
    values = value if isinstance(value, list) else str(value).splitlines()
    clean: list[dict[str, object]] = []
    seen: set[str] = set()
    for value_item in values:
        item = _sanitize_memory_item(value_item, memory_type)
        if item is None:
            continue
        key = str(item["text"]).casefold()
        if key in seen or _is_expired(item):
            continue
        seen.add(key)
        clean.append(item)
        if len(clean) >= MAX_LIST_ITEMS:
            break
    return _rank_memory_items(clean)


def _sanitize_memory_item(value: object, memory_type: str) -> dict[str, object] | None:
    if isinstance(value, dict):
        text = _sanitize_optional_text(value.get("text"))
        if not text:
            return None
        defaults = MEMORY_DEFAULTS[memory_type]
        created_at = _sanitize_timestamp(value.get("created_at")) or _now_iso()
        updated_at = _sanitize_timestamp(value.get("updated_at")) or created_at
        return {
            "text": text[:MAX_ITEM_LENGTH],
            "type": memory_type,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_used_at": _sanitize_timestamp(value.get("last_used_at")),
            "use_count": _sanitize_int(value.get("use_count"), default=0),
            "confidence": _sanitize_float(
                value.get("confidence"),
                default=DEFAULT_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
            ),
            "importance": _sanitize_float(
                value.get("importance"),
                default=float(defaults["importance"]),
                minimum=0.0,
                maximum=1.0,
            ),
            "decay_rate": _sanitize_float(
                value.get("decay_rate"),
                default=float(defaults["decay_rate"]),
                minimum=0.0,
                maximum=1.0,
            ),
            "expires_at": _sanitize_timestamp(value.get("expires_at")),
        }
    text = _sanitize_optional_text(value)
    if not text:
        return None
    return _new_memory_item(text, memory_type)


def _new_memory_item(text: str, memory_type: str) -> dict[str, object]:
    defaults = MEMORY_DEFAULTS[memory_type]
    now = _now_iso()
    return {
        "text": text[:MAX_ITEM_LENGTH],
        "type": memory_type,
        "created_at": now,
        "updated_at": now,
        "last_used_at": None,
        "use_count": 0,
        "confidence": DEFAULT_CONFIDENCE,
        "importance": defaults["importance"],
        "decay_rate": defaults["decay_rate"],
        "expires_at": None,
    }


def _rank_memory_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    active = [item for item in items if not _is_expired(item)]
    return sorted(active, key=_memory_score, reverse=True)


def _memory_score(item: dict[str, object]) -> float:
    importance = float(item.get("importance", 0.0))
    confidence = float(item.get("confidence", 0.0))
    use_count = int(item.get("use_count", 0))
    decay_rate = float(item.get("decay_rate", 0.0))
    age_days = _age_days(str(item.get("updated_at") or item.get("created_at") or _now_iso()))
    reinforcement_boost = min(use_count * 0.05, MAX_REINFORCEMENT_BOOST)
    return importance + confidence + reinforcement_boost - (age_days * decay_rate)


def _reinforce_item(item: dict[str, object]) -> None:
    _mark_item_used(item)
    item["confidence"] = min(
        1.0,
        float(item.get("confidence", DEFAULT_CONFIDENCE)) + REINFORCED_CONFIDENCE_STEP,
    )
    item["importance"] = min(
        1.0,
        float(item.get("importance", 0.0)) + REINFORCED_IMPORTANCE_STEP,
    )


def _mark_item_used(item: dict[str, object]) -> None:
    now = _now_iso()
    item["last_used_at"] = now
    item["updated_at"] = now
    item["use_count"] = int(item.get("use_count", 0)) + 1


def _is_expired(item: dict[str, object]) -> bool:
    expires_at = item.get("expires_at")
    if not expires_at:
        return False
    parsed = _parse_timestamp(str(expires_at))
    return parsed is not None and parsed <= datetime.now(timezone.utc)


def _merge_unique(existing: object, incoming: object) -> list[str]:
    merged: list[str] = []
    for item in [*_sanitize_list(existing), *_sanitize_list(incoming)]:
        if item not in merged:
            merged.append(item)
        if len(merged) >= MAX_LIST_ITEMS:
            break
    return merged


def _sanitize_timestamp(value: object) -> str | None:
    if value is None:
        return None
    parsed = _parse_timestamp(str(value))
    return parsed.isoformat() if parsed else None


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_days(timestamp: str) -> float:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400)


def _sanitize_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, minimum), maximum)


def _sanitize_int(value: object, *, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(0, number)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
