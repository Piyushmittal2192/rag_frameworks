import hashlib

from rag_framework.models import DocumentChunk


def split_documents(
    documents: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in documents:
        normalized = " ".join(text.split())
        start = 0
        chunk_number = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(f"{source}:{chunk_number}:{chunk_text}".encode()).hexdigest()[:16]
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        source=source,
                        metadata={"chunk": chunk_number},
                    )
                )
                chunk_number += 1
            if end == len(normalized):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
