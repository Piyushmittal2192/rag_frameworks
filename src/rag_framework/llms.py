import re
from typing import Protocol

import httpx


class LLM(Protocol):
    async def generate(self, prompt: str) -> str:
        ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", "")).strip()


class OpenAICompatibleLLM:
    """Adapter for open-source OpenAI-compatible servers, such as vLLM or llama.cpp."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"

    async def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()


class GitHubModelsLLM:
    """Adapter for GitHub Models chat completions."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str = "https://models.github.ai/inference",
    ) -> None:
        if not api_key:
            raise ValueError("GitHub Models requires GITHUB_TOKEN or LLM_API_KEY.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str) -> str:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()


class EchoLLM:
    """Deterministic local demo LLM used when no external model server is running."""

    async def generate(self, prompt: str) -> str:
        if "Break the question into focused search sub-questions" in prompt:
            return _decompose_question(prompt)
        if "Return only \"relevant\" or \"irrelevant\"" in prompt:
            return _grade_relevance(prompt)
        if "Rewrite the question into a better search query" in prompt:
            return _extract_question(prompt)
        return _answer_from_context(prompt)


def create_llm(provider: str, model: str, base_url: str, api_key: str | None = None) -> LLM:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaLLM(model=model, base_url=base_url)
    if provider in {"openai-compatible", "openai_compatible", "vllm", "llamacpp", "lmstudio"}:
        return OpenAICompatibleLLM(model=model, base_url=base_url, api_key=api_key)
    if provider in {"github-models", "github_models", "github"}:
        github_base_url = (
            base_url
            if base_url and base_url != "http://localhost:11434"
            else "https://models.github.ai/inference"
        )
        return GitHubModelsLLM(model=model, api_key=api_key, base_url=github_base_url)
    if provider == "echo":
        return EchoLLM()
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _extract_question(prompt: str) -> str:
    question_match = re.search(r"Question:\s*(.+?)(?:\n\n|\nSearch query:|$)", prompt, re.DOTALL)
    if not question_match:
        return ""
    return " ".join(question_match.group(1).split())


def _extract_context(prompt: str) -> str:
    context_match = re.search(r"Context:\s*(.+?)\n\nAnswer", prompt, re.DOTALL)
    if not context_match:
        context_match = re.search(r"Context:\s*(.+)$", prompt, re.DOTALL)
    if not context_match:
        return ""
    return context_match.group(1).strip()


def _answer_from_context(prompt: str) -> str:
    question = _extract_question(prompt)
    context = _extract_context(prompt)
    if not context or context == "No retrieved context.":
        return "I do not have enough retrieved context to answer that."
    if _asks_for_absence(question):
        return "The retrieved context does not state which open source LLM interfaces are not supported."

    source_match = re.search(r"\[(.+?): chunk (\d+) \| score", context)
    source_label = ""
    if source_match:
        source_label = f" [{source_match.group(1)}: chunk {source_match.group(2)}]"

    plain_context = re.sub(r"\[.+?: chunk \d+ \| score [0-9.]+\]\s*", "", context)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", plain_context) if sentence.strip()]
    question_terms = _keywords(question)
    selected = [
        sentence
        for _, sentence in sorted(
            (
                (len(_keywords(sentence).intersection(question_terms)), sentence)
                for sentence in sentences
            ),
            reverse=True,
        )
        if question_terms and _keywords(sentence).intersection(question_terms)
    ]
    if not selected:
        selected = sentences[:2]

    answer = " ".join(selected[:3]).strip()
    if source_label:
        answer = f"{answer}{source_label}"
    return answer


def _grade_relevance(prompt: str) -> str:
    question = _extract_question(prompt)
    context = _extract_context(prompt)
    if _keywords(question).intersection(_keywords(context)):
        return "relevant"
    return "irrelevant"


def _decompose_question(prompt: str) -> str:
    question = _extract_question(prompt)
    parts = re.split(r"\s+(?:and|versus|vs\.?|compare|contrast)\s+", question, flags=re.IGNORECASE)
    sub_questions = [part.strip(" ?.") for part in parts if part.strip(" ?.")]
    if len(sub_questions) <= 1:
        return question
    return "\n".join(f"{index}. {part}?" for index, part in enumerate(sub_questions[:4], start=1))


def _keywords(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]*", text.lower())
        if token not in stopwords and len(token) > 2
    }


def _asks_for_absence(question: str) -> bool:
    lowered = f" {question.lower()} "
    absence_markers = (" not ", " unsupported ", "not supported", "doesn't support", "do not support")
    return any(marker in lowered for marker in absence_markers)


def _normalize_token(token: str) -> str:
    token = token.replace("-", "")
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token
