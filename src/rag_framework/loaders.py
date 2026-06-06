from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents(path: Path) -> list[tuple[str, str]]:
    """Return (source, text) pairs from a file or directory."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    documents: list[tuple[str, str]] = []
    for file_path in files:
        text = _load_file(file_path)
        if text.strip():
            documents.append((str(file_path), text))
    return documents


def _load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")
