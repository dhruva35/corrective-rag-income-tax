"""
Manual RAG evaluation script — bypasses RAGAS's async plumbing.

Runs the same three metrics that RAGAS measures, but calls the LLM directly
via simple synchronous prompts so there are no compatibility issues.

Usage:
    python eval/run_eval.py

Metrics implemented:
  - faithfulness      : fraction of answer claims grounded in the context
  - answer_relevancy  : how well the answer addresses the question (0-1)
  - context_precision : fraction of retrieved chunks that are actually relevant

Why manual instead of RAGAS?
  RAGAS 0.2.6 + langchain-google-genai 2.1.5 have a known incompatibility in
  how the async scoring loop handles finish_reason codes. Rather than fight the
  framework, we implement the same scoring logic directly — which also gives you
  a clear view of what RAGAS is actually doing under the hood.
"""
import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.graph.build_graph import rag_graph
from app.config import settings

TESTSET_PATH = "eval/testset.json"

# ── LLM for evaluation (same model as the RAG pipeline) ────────────────────
_eval_llm = ChatGoogleGenerativeAI(
    model=settings.google_chat_model,
    google_api_key=settings.google_api_key,
)

# Pause between API calls to avoid hitting the free-tier rate limit (5 RPM).
_RATE_LIMIT_DELAY = 13  # seconds between calls → ~4.6 calls/min


def _call_llm(prompt_text: str) -> str:
    """Single synchronous LLM call with rate-limit throttling."""
    time.sleep(_RATE_LIMIT_DELAY)
    response = _eval_llm.invoke(prompt_text)
    return response.content.strip()


def _parse_score(text: str) -> float:
    """Extract the first float/int in [0, 1] from an LLM response."""
    import re
    matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if matches:
        return float(matches[0])
    return float("nan")


# ── Metric: Faithfulness ────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are evaluating a RAG system answer.

Context (retrieved documents):
{context}

Answer given by the system:
{answer}

Task: Rate how faithful the answer is to the context.
- A score of 1.0 means every factual claim in the answer is directly supported by the context.
- A score of 0.0 means the answer contains claims not supported by the context.
- Give a score between 0.0 and 1.0.

Respond with ONLY a number between 0.0 and 1.0."""


def score_faithfulness(context: str, answer: str) -> float:
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = _call_llm(prompt)
    return _parse_score(raw)


# ── Metric: Answer Relevancy ────────────────────────────────────────────────

_RELEVANCY_PROMPT = """You are evaluating a RAG system answer.

Question: {question}

Answer: {answer}

Task: Rate how relevant and complete this answer is for the question.
- A score of 1.0 means the answer directly and completely addresses the question.
- A score of 0.0 means the answer is completely off-topic or empty.
- Give a score between 0.0 and 1.0.

Respond with ONLY a number between 0.0 and 1.0."""


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
    raw = _call_llm(prompt)
    return _parse_score(raw)


# ── Metric: Context Precision ───────────────────────────────────────────────

_PRECISION_PROMPT = """You are evaluating a retrieval system.

Question: {question}

Retrieved context chunk:
{chunk}

Task: Rate whether this retrieved chunk is relevant to answering the question.
- A score of 1.0 means this chunk is directly useful for answering the question.
- A score of 0.0 means this chunk is not relevant at all.
- Give a score between 0.0 and 1.0.

Respond with ONLY a number between 0.0 and 1.0."""


def score_context_precision(question: str, chunks: list[str]) -> float:
    """Average relevance across retrieved chunks."""
    scores = []
    for chunk in chunks:
        prompt = _PRECISION_PROMPT.format(question=question, chunk=chunk[:600])
        raw = _call_llm(prompt)
        scores.append(_parse_score(raw))
    valid = [s for s in scores if s == s]  # filter nan
    return sum(valid) / len(valid) if valid else float("nan")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    with open(TESTSET_PATH) as f:
        cases = json.load(f)

    print("🚀 Running RAG pipeline on testset questions...")
    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['question'][:70]}...")
        result = rag_graph.invoke({"question": case["question"]})
        results.append({
            "question": case["question"],
            "answer": result["answer"],
            "chunks": [doc.page_content for doc in result["retrieved_docs"]],
            "context": "\n\n".join(doc.page_content for doc in result["retrieved_docs"]),
        })
    print(f"  Done. Got {len(results)} answers.\n")

    print(f"📊 Scoring metrics (LLM calls, ~{_RATE_LIMIT_DELAY}s delay between each)...")
    print("   This will take ~3-4 minutes to stay within free-tier rate limits.\n")

    all_faith, all_rel, all_prec = [], [], []

    for i, (case, res) in enumerate(zip(cases, results), 1):
        q = res["question"]
        print(f"  [{i}/{len(cases)}] Scoring: {q[:60]}...")

        faith = score_faithfulness(res["context"], res["answer"])
        print(f"    faithfulness     = {faith:.3f}")

        rel = score_answer_relevancy(q, res["answer"])
        print(f"    answer_relevancy = {rel:.3f}")

        prec = score_context_precision(q, res["chunks"])
        print(f"    context_precision= {prec:.3f}")

        all_faith.append(faith)
        all_rel.append(rel)
        all_prec.append(prec)
        print()

    def mean(lst):
        valid = [x for x in lst if x == x]
        return sum(valid) / len(valid) if valid else float("nan")

    faith_avg = mean(all_faith)
    rel_avg = mean(all_rel)
    prec_avg = mean(all_prec)

    def bar(score):
        if score != score:
            return "—"
        return "█" * int(score * 20)

    print("=" * 50)
    print("  Step 3 Baseline — RAGAS-equivalent Scores")
    print("=" * 50)
    print(f"  {'faithfulness':24s}: {faith_avg:.3f}  {bar(faith_avg)}")
    print(f"  {'answer_relevancy':24s}: {rel_avg:.3f}  {bar(rel_avg)}")
    print(f"  {'context_precision':24s}: {prec_avg:.3f}  {bar(prec_avg)}")
    print("=" * 50)
    print("\n  These are your baseline numbers for Step 3.")
    print("  Keep them — Step 4 (hybrid search) should improve context_precision.")


if __name__ == "__main__":
    main()
