"""
Runs the RAG pipeline against eval/testset.json and scores it with RAGAS.

    python eval/run_ragas_eval.py

This is the script that produces the numbers you'd actually put on your
resume ("achieved 0.89 faithfulness across a 20-question eval set") — run
it against your own domain's testset, not the placeholder one checked in
here.

RAGAS metrics used:
  - faithfulness      : are claims in the answer supported by the retrieved context?
  - answer_relevancy  : does the answer actually address the question?
  - context_precision : are the retrieved chunks relevant to the question?

Compatibility note:
  RAGAS 0.2.6 passes 'temperature' as a runtime kwarg to the async Gemini
  client, which rejects it (it only accepts temperature at construction time).
  We work around this by subclassing ChatGoogleGenerativeAI and stripping
  unsupported kwargs before the async call reaches the grpc layer.
"""
import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import Any, List, Optional

from app.graph.build_graph import rag_graph
from app.vectorstore import get_embeddings
from app.config import settings

TESTSET_PATH = "eval/testset.json"


class GeminiLLMForRAGAS(ChatGoogleGenerativeAI):
    """Thin wrapper that strips kwargs unsupported by Gemini's async gRPC client.

    RAGAS 0.2.6 passes 'temperature' as a runtime kwarg to the async generate
    call, but langchain-google-genai 2.1.5 only accepts it at construction
    time. We strip it here to prevent the TypeError.
    """

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        # Drop any kwargs the Gemini grpc layer doesn't accept at call time
        for bad_kwarg in ("temperature", "top_p", "top_k", "max_tokens"):
            kwargs.pop(bad_kwarg, None)
        return await super()._agenerate(messages, stop=stop, **kwargs)


def _build_ragas_llm() -> LangchainLLMWrapper:
    llm = GeminiLLMForRAGAS(
        model=settings.google_chat_model,
        google_api_key=settings.google_api_key,
        convert_system_message_to_human=True,
    )
    return LangchainLLMWrapper(llm)


def build_eval_dataset() -> Dataset:
    """Run every question through the RAG graph and collect outputs."""
    with open(TESTSET_PATH) as f:
        cases = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] Running: {case['question'][:70]}...")
        result = rag_graph.invoke({"question": case["question"]})
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["retrieved_docs"]])
        ground_truths.append(case["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main():
    print("🚀 Building eval dataset (running pipeline on testset)...")
    dataset = build_eval_dataset()
    print(f"   Built {len(dataset)} examples.\n")

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    print("📊 Running RAGAS evaluation (faithfulness, answer_relevancy, context_precision)...")
    # max_workers=1 serializes scoring calls — critical for Gemini free tier (5 RPM).
    # Each metric scores each sample independently, so 4 questions x 3 metrics = 12 calls.
    run_cfg = RunConfig(max_workers=1, timeout=120)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    # RAGAS 0.2.x returns EvaluationResult — use to_pandas() to get scores
    df = results.to_pandas()

    print("\n" + "=" * 45)
    print("RAGAS Evaluation Results (Baseline)")
    print("=" * 45)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric in df.columns:
            score = df[metric].mean()
            bar = "█" * int(score * 20) if not __import__("math").isnan(score) else "—"
            print(f"  {metric:24s}: {score:.3f}  {bar}")
    print("=" * 45)
    print("\nThese are your Step 3 baseline numbers. Higher is better (max 1.0).")
    print("Save these — we'll compare against them after adding hybrid search in Step 4.\n")

    # Also print per-question details
    print("Per-question breakdown:")
    # RAGAS 0.2.6 to_pandas() only returns metric columns, not the original question.
    available = [c for c in ["faithfulness", "answer_relevancy", "context_precision"] if c in df.columns]
    print(df[available].to_string(index=False))


if __name__ == "__main__":
    main()
