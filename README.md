# RAG Framework

A deployable Python framework for three retrieval-augmented generation pipelines:

- **Standard RAG**: retrieve the top matching chunks, pass them to an LLM, and answer with sources.
- **Corrective RAG**: retrieve, grade/filter weak context, rewrite the query when evidence is poor, retrieve again, then answer from the corrected context.
- **Planner RAG**: dynamically decompose broad questions into focused retrieval tasks, retrieve evidence for each task, fuse the evidence, then synthesize one answer.
- **Hybrid retrieval**: combine dense vector search and BM25 lexical search with Reciprocal Rank Fusion.

The LLM layer uses open-source-friendly interfaces:

- `ollama` for local Ollama models.
- `openai-compatible` for vLLM, llama.cpp server, LM Studio, Text Generation Inference gateways, and similar local/open-source servers.
- `github-models` for GitHub Models hosted chat completions.

## Documentation

For learning guides, architecture notes, ML design, and pipeline diagrams, see [documentation/README.md](documentation/README.md).

## Project Layout

```text
src/rag_framework/
  app.py            FastAPI service
  cli.py            ingest and ask commands
  pipelines.py      standard, corrective, and planner RAG
  llms.py           Ollama and OpenAI-compatible adapters
  vector_store.py   persisted vector + BM25 hybrid retriever
  loaders.py        txt, md, pdf loading
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Add documents to `data/docs` as `.txt`, `.md`, or `.pdf`, then build the index:

```bash
rag-framework ingest
```

Run with Ollama:

```bash
ollama pull llama3.1:8b
ollama serve
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:8b
export LLM_BASE_URL=http://localhost:11434
```

Ask from the CLI:

```bash
rag-framework ask "What does this project do?"
rag-framework ask "What does this project do?" --pipeline corrective
rag-framework ask "Compare vector retrieval, BM25, and reranking" --pipeline planner
```

Start the API:

```bash
uvicorn rag_framework.app:app --host 0.0.0.0 --port 8000
```

Open the web interface:

```text
http://localhost:8000/
```

The interface includes:

- Standard RAG, Corrective RAG, and Planner RAG flow diagrams.
- Step-by-step pipeline trace cards.
- Run observability with trace ID, provider, model, total duration, source count, scores, and correction status.
- Source chunks with RRF scores, vector/BM25 ranks, and raw retrieval scores.

## Hybrid Retrieval

Each query runs two retrievers:

```text
question
  -> dense vector search
  -> BM25 keyword search
  -> Reciprocal Rank Fusion
  -> lightweight cross-encoder reranker
  -> final ranked context
```

RRF avoids mixing incompatible raw score scales. A chunk gets a small score boost from each retriever list where it appears:

```text
rrf_score = sum(1 / (60 + rank))
```

Chunks found by both vector search and BM25 naturally move higher in the final ranking.

When `ENABLE_RERANKER=true`, the app reranks the RRF candidate set with a lightweight cross-encoder:

```bash
export ENABLE_RERANKER=true
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker directly scores query/chunk relevance after hybrid retrieval, then returns the final top K chunks.

## Faithfulness Judge

The app can run an optional secondary LLM-as-judge after answer generation. The judge receives the original question, retrieved context, and generated answer, then returns a faithfulness verdict:

```text
generated answer
  -> judge against retrieved context
  -> grounded / partially_grounded / unsupported
  -> unsupported claims + citation issues
```

Enable it with:

```bash
export ENABLE_LLM_JUDGE=true
export JUDGE_PROVIDER=github-models
export JUDGE_MODEL=meta/meta-llama-3.1-8b-instruct
export JUDGE_BASE_URL=https://models.github.ai/inference
```

If `JUDGE_PROVIDER` or `JUDGE_MODEL` are omitted, the judge falls back to the main LLM provider and model. The UI shows the judge verdict in the answer panel, run metadata, execution timeline, and Faithfulness Review panel.

LLM judges are a useful trust signal, not a mathematical proof. For production use, combine judge scores with retrieval evals, citation checks, and curated regression datasets.

## Planner RAG

Planner RAG is useful for broader or multi-part questions where one search query is likely too compressed:

```text
original question
  -> decompose into focused sub-questions
  -> retrieve per sub-question with vector + BM25 + RRF
  -> fuse sub-query evidence
  -> optional global reranking
  -> synthesize one cited answer
```

The decomposition step is intentionally bounded. It returns at most four sub-questions, and if the original question is already focused, the planner can keep it as a single retrieval task. The trace shows the generated sub-questions, each sub-query retrieval, and the final fused evidence set.

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare vector retrieval, BM25, and reranking","pipeline":"planner"}'
```

## Corrective RAG Quality Gates

Corrective RAG does not blindly trust rewritten queries. A rewrite is accepted only if it passes two gates:

```text
original query
  -> weak evidence detected
  -> rewrite query
  -> intent preservation gate
  -> second retrieval + reranking
  -> evidence improvement gate
  -> accept rewrite or fall back
```

The intent preservation gate checks:

- embedding similarity between the original and rewritten query
- important-term preservation
- polarity preservation, such as not changing "not supported" into "supported"

The evidence improvement gate checks whether the rewritten query improves retrieved evidence after hybrid retrieval and reranking. If the rewrite drifts or does not improve evidence enough, the system rejects it and falls back to the initial retrieval results.

Correction is triggered when any of these are true:

- no chunks are kept
- mean vector relevance is below `RELEVANCE_THRESHOLD`
- reranker evidence is below `RERANKER_EVIDENCE_THRESHOLD` when the reranker is enabled

Relevant settings:

```bash
export REWRITE_INTENT_THRESHOLD=0.65
export REWRITE_EVIDENCE_MARGIN=0.05
export RERANKER_EVIDENCE_THRESHOLD=0.0
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this project do?","pipeline":"corrective"}'
```

## OpenAI-Compatible Local Servers

For vLLM, llama.cpp server, or LM Studio:

```bash
export LLM_PROVIDER=openai-compatible
export LLM_MODEL=your-model-name
export LLM_BASE_URL=http://localhost:8001
export LLM_API_KEY=not-needed
```

The adapter calls `POST /v1/chat/completions`.

## GitHub Models

GitHub Models can run hosted models through a chat-completions endpoint. Create a token with `models:read`, then configure:

```bash
export LLM_PROVIDER=github-models
export LLM_MODEL=publisher/model-from-github-catalog
export GITHUB_TOKEN=your-github-token
```

By default, the adapter calls:

```text
https://models.github.ai/inference/chat/completions
```

You can override the base URL with `LLM_BASE_URL`.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Important settings:

- `DOCS_DIR`: directory with source documents.
- `INDEX_DIR`: persisted vector index directory.
- `EMBEDDING_MODEL`: Sentence Transformers model.
- `LLM_PROVIDER`: `ollama`, `openai-compatible`, `github-models`, or `echo`.
- `LLM_MODEL`: model name for the selected provider.
- `GITHUB_TOKEN`: token for GitHub Models when using `LLM_PROVIDER=github-models`.
- `RELEVANCE_THRESHOLD`: score floor used by corrective RAG.
- `ENABLE_LLM_JUDGE`: run a secondary judge after answer generation.
- `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_BASE_URL`: optional independent judge model settings.

## Docker

Build:

```bash
docker build -t rag-framework .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -e LLM_PROVIDER=ollama \
  -e LLM_MODEL=llama3.1:8b \
  -e LLM_BASE_URL=http://host.docker.internal:11434 \
  rag-framework
```

Build the index before starting the API, or run `rag-framework ingest` inside the container with `data/docs` mounted.

## Tests

```bash
pytest
```
