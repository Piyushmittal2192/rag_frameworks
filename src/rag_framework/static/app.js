const form = document.querySelector("#query-form");
const questionInput = document.querySelector("#question");
const topKInput = document.querySelector("#top-k");
const submitButton = document.querySelector("#submit-button");
const answerTitle = document.querySelector("#answer-title");
const answer = document.querySelector("#answer");
const pipelineBadge = document.querySelector("#pipeline-badge");
const observability = document.querySelector("#observability");
const steps = document.querySelector("#steps");
const sources = document.querySelector("#sources");
const pipelineInputs = document.querySelectorAll('input[name="pipeline"]');
const defaultQuestionButtons = document.querySelectorAll(".default-question");

const defaultQuestions = {
  standard: "What open source LLM interfaces are supported?",
  corrective: "How do I build the search files?",
  planner: "Compare Standard RAG, Corrective RAG, and Planner RAG.",
};

pipelineInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) {
      setPipelineQuestion(input.value);
    }
  });
});

defaultQuestionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const pipeline = button.dataset.pipeline;
    const question = button.dataset.question;
    if (!pipeline || !question) {
      return;
    }
    const pipelineInput = document.querySelector(`input[name="pipeline"][value="${pipeline}"]`);
    if (pipelineInput) {
      pipelineInput.checked = true;
    }
    setPipelineQuestion(pipeline, question);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const pipeline = new FormData(form).get("pipeline");
  const question = questionInput.value.trim();
  const topK = Number(topKInput.value || 5);

  if (!question) {
    setError("Enter a question before running the pipeline.");
    return;
  }

  setLoading(pipeline);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, pipeline, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    renderResult(await response.json());
  } catch (error) {
    setError(error.message || "The query failed.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Run";
  }
});

function setLoading(pipeline) {
  submitButton.disabled = true;
  submitButton.textContent = "Running";
  answerTitle.textContent = "Running";
  pipelineBadge.textContent = pipeline;
  answer.textContent = "Retrieving context...";
  observability.innerHTML = '<div class="empty-state">Trace running...</div>';
  steps.innerHTML = "";
  sources.innerHTML = "";
}

function setPipelineQuestion(pipeline, question = defaultQuestions[pipeline]) {
  if (!question) {
    return;
  }
  questionInput.value = question;
  pipelineBadge.textContent = pipeline;
}

function setError(message) {
  answerTitle.textContent = "Error";
  pipelineBadge.textContent = "stopped";
  answer.textContent = message;
  observability.innerHTML = "";
  steps.innerHTML = "";
  sources.innerHTML = "";
}

function renderResult(result) {
  answerTitle.textContent = getAnswerTitle(result);
  pipelineBadge.textContent = result.pipeline;
  answer.textContent = result.answer || "No answer returned.";
  renderObservability(result.trace);
  renderSteps(result.steps || []);
  renderSources(result.sources || []);
}

function getAnswerTitle(result) {
  if (result.pipeline === "planner") {
    return "Planned Answer";
  }
  if (result.correction_applied) {
    return "Corrected Answer";
  }
  return "Answer";
}

function renderObservability(trace) {
  if (!trace) {
    observability.innerHTML = '<div class="empty-state">No trace returned.</div>';
    return;
  }

  const metrics = [
    ["Trace ID", trace.trace_id],
    ["Provider", trace.llm_provider],
    ["Model", trace.llm_model],
    ["Total Time", `${formatNumber(trace.total_duration_ms)} ms`],
    ["Top K", trace.top_k],
    ["Sources", trace.source_count],
    ["Best Score", formatNumber(trace.best_score)],
    ["Mean Score", formatNumber(trace.mean_score)],
    ["Correction", trace.correction_applied ? "Applied" : "Not applied"],
    ["Reranker", trace.reranker_enabled ? "Enabled" : "Disabled"],
  ];
  if (trace.reranker_model) {
    metrics.push(["Reranker Model", trace.reranker_model]);
  }

  observability.innerHTML = metrics
    .map(
      ([label, value]) => `
        <div class="metric">
          <span class="metric-label">${escapeHtml(String(label))}</span>
          <span class="metric-value">${escapeHtml(String(value))}</span>
        </div>
      `
    )
    .join("");
}

function renderSteps(items) {
  if (!items.length) {
    steps.innerHTML = '<li class="empty-state">No pipeline trace returned.</li>';
    return;
  }

  steps.innerHTML = items
    .map((step, index) => {
      const status = escapeHtml(step.status || "completed");
      return `
        <li class="step ${status}">
          <div class="step-title-row">
            <h3>${index + 1}. ${escapeHtml(step.name)}</h3>
            <span class="status">${status}</span>
          </div>
          <span class="step-duration">${formatNumber(step.duration_ms || 0)} ms</span>
          <p>${escapeHtml(step.description)}</p>
          ${renderDetails(step.details || {})}
        </li>
      `;
    })
    .join("");
}

function renderDetails(details) {
  const entries = Object.entries(details);
  if (!entries.length) {
    return "";
  }

  const pills = entries
    .map(([key, value]) => {
      const label = key.replaceAll("_", " ");
      const shownValue = Array.isArray(value) ? value.join(", ") : value;
      return `<span class="detail-pill">${escapeHtml(label)}: ${escapeHtml(String(shownValue))}</span>`;
    })
    .join("");

  return `<div class="detail-list">${pills}</div>`;
}

function renderSources(items) {
  if (!items.length) {
    sources.innerHTML = '<div class="empty-state">No sources returned.</div>';
    return;
  }

  sources.innerHTML = items
    .map((item) => {
      const chunk = item.chunk;
      const chunkNumber = chunk.metadata?.chunk ?? 0;
      const retrievers = item.retrieval_sources?.join(" + ") || "unknown";
      const ranks = Object.entries(item.ranks || {})
        .map(([name, rank]) => `${name} #${rank}`)
        .join(", ");
      const rawScores = Object.entries(item.raw_scores || {})
        .map(([name, score]) => `${name}: ${formatNumber(score)}`)
        .join(", ");
      return `
        <article class="source">
          <h3>${escapeHtml(chunk.source)} · chunk ${escapeHtml(String(chunkNumber))}</h3>
          <span class="source-score">
            Final ${formatNumber(item.score)} · RRF ${formatNumber(item.raw_scores?.rrf || 0)} · ${escapeHtml(retrievers)}
          </span>
          ${ranks ? `<div class="source-meta">${escapeHtml(ranks)}</div>` : ""}
          ${rawScores ? `<div class="source-meta">${escapeHtml(rawScores)}</div>` : ""}
          <p>${escapeHtml(chunk.text)}</p>
        </article>
      `;
    })
    .join("");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  const number = Number(value || 0);
  if (number >= 100) {
    return number.toFixed(0);
  }
  if (number >= 10) {
    return number.toFixed(1);
  }
  return number.toFixed(3);
}
