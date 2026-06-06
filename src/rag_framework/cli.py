from pathlib import Path

import typer

from rag_framework.config import get_settings
from rag_framework.embeddings import SentenceTransformerEmbedder
from rag_framework.judges import LLMFaithfulnessJudge
from rag_framework.llms import create_llm
from rag_framework.loaders import load_documents
from rag_framework.pipelines import CorrectiveRAGPipeline, PlannerRAGPipeline, StandardRAGPipeline
from rag_framework.rerankers import CrossEncoderReranker
from rag_framework.splitter import split_documents
from rag_framework.vector_store import VectorStore

app = typer.Typer(help="Build and query standard RAG and corrective RAG pipelines.")


@app.command()
def ingest(
    docs: Path | None = typer.Option(None, help="File or directory containing .txt, .md, or .pdf docs."),
    index: Path | None = typer.Option(None, help="Directory where the vector index will be stored."),
) -> None:
    settings = get_settings()
    docs_path = docs or settings.docs_dir
    index_path = index or settings.index_dir
    documents = load_documents(docs_path)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore.build(chunks, embedder)
    store.save(index_path)
    typer.echo(f"Indexed {len(chunks)} chunks from {len(documents)} documents into {index_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    pipeline: str = typer.Option("standard", help="standard, corrective, or planner"),
    index: Path | None = typer.Option(None, help="Directory containing the vector index."),
) -> None:
    import asyncio

    asyncio.run(_ask(question=question, pipeline_name=pipeline, index=index))


async def _ask(question: str, pipeline_name: str, index: Path | None) -> None:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore.load(index or settings.index_dir)
    llm = create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or settings.github_token,
    )
    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.enable_reranker
        else None
    )
    judge_llm = None
    if settings.enable_llm_judge:
        judge_llm = create_llm(
            provider=settings.judge_provider or settings.llm_provider,
            model=settings.judge_model or settings.llm_model,
            base_url=settings.judge_base_url or settings.llm_base_url,
            api_key=settings.judge_api_key or settings.llm_api_key or settings.github_token,
        )
    judge = LLMFaithfulnessJudge(judge_llm) if judge_llm is not None else None
    if pipeline_name == "corrective":
        pipeline = CorrectiveRAGPipeline(
            store,
            embedder,
            llm,
            top_k=settings.top_k,
            relevance_threshold=settings.relevance_threshold,
            rewrite_intent_threshold=settings.rewrite_intent_threshold,
            rewrite_evidence_margin=settings.rewrite_evidence_margin,
            reranker_evidence_threshold=settings.reranker_evidence_threshold,
            reranker=reranker,
            judge=judge,
        )
    elif pipeline_name == "planner":
        pipeline = PlannerRAGPipeline(
            store,
            embedder,
            llm,
            top_k=settings.top_k,
            reranker=reranker,
            judge=judge,
        )
    else:
        pipeline = StandardRAGPipeline(
            store,
            embedder,
            llm,
            top_k=settings.top_k,
            reranker=reranker,
            judge=judge,
        )

    result = await pipeline.answer(question)
    typer.echo(result.answer)
    typer.echo("\nSources:")
    for source in result.sources:
        typer.echo(f"- {source.chunk.source} chunk {source.chunk.metadata.get('chunk', 0)} ({source.score:.3f})")
    if result.judge:
        typer.echo(
            f"\nFaithfulness: {result.judge.verdict} "
            f"({result.judge.faithfulness_score:.2f})"
        )


if __name__ == "__main__":
    app()
