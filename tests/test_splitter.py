from rag_framework.splitter import split_documents


def test_split_documents_preserves_source_and_creates_chunks():
    chunks = split_documents(
        [("doc.md", "alpha beta gamma " * 20)],
        chunk_size=40,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk.source == "doc.md" for chunk in chunks)
    assert chunks[0].metadata["chunk"] == 0
