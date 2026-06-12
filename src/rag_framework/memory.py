import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from rag_framework.models import PersonalizationContext

MemoryMode = Literal["stateless", "stateful"]

MAX_KEY_LENGTH = 40
MAX_VALUE_LENGTH = 220


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
    ) -> PersonalizationContext:
        session = _sanitize_preferences(session_preferences or {})
        if mode == "stateless":
            return PersonalizationContext(
                mode=mode,
                user_id=None,
                preferences=session,
                session_preferences=session,
                memory_loaded=False,
                memory_saved=False,
            )

        clean_user_id = _sanitize_user_id(user_id)
        if not clean_user_id:
            raise ValueError("Stateful memory requires a user_id.")

        store = self._read_store()
        user_memory = store.get(clean_user_id, {})
        persisted = _sanitize_preferences(user_memory.get("preferences", {}))
        preferences = {**persisted, **session}
        memory_saved = False

        if remember_preferences and session:
            store[clean_user_id] = {
                "preferences": preferences,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_store(store)
            memory_saved = True

        return PersonalizationContext(
            mode=mode,
            user_id=clean_user_id,
            preferences=preferences,
            persisted_preferences=persisted,
            session_preferences=session,
            memory_loaded=bool(persisted),
            memory_saved=memory_saved,
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
    if context is None or not context.preferences:
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
    for key, value in sorted(context.preferences.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def summarize_preferences(context: PersonalizationContext | None) -> list[str]:
    if context is None:
        return []
    return [f"{key}: {value}" for key, value in sorted(context.preferences.items())]


def _sanitize_preferences(preferences: dict[str, object]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in preferences.items():
        clean_key = _sanitize_key(str(key))
        clean_value = " ".join(str(value).split())[:MAX_VALUE_LENGTH]
        if clean_key and clean_value:
            clean[clean_key] = clean_value
    return clean


def _sanitize_key(key: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", key.strip().lower())
    return normalized.strip("_")[:MAX_KEY_LENGTH]


def _sanitize_user_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9_.@-]+", "-", user_id.strip())
    return normalized[:80] or None
