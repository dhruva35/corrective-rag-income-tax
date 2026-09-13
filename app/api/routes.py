"""
FastAPI routes for the RAG assistant.

Endpoints:
  GET  /health          — liveness probe
  GET  /pipeline-info   — describes the current graph architecture
  POST /query           — synchronous Q&A, returns full answer + pipeline trace
  POST /stream          — streaming Q&A via Server-Sent Events (SSE)
  POST /ingest          — trigger a re-index of source documents
"""
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph.build_graph import rag_graph
from app.ingestion.pipeline import run_ingestion

router = APIRouter()


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class DocGrade(BaseModel):
    source: str
    grade: str   # "yes" or "no"


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[str]          # de-duplicated list of relevant source files
    doc_grades: list[DocGrade]  # per-chunk grading result (pipeline transparency)


class IngestResponse(BaseModel):
    chunks_indexed: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_graph(question: str) -> dict:
    """Invoke the RAG graph and return the final state dict."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return rag_graph.invoke({"question": question})


def _extract_response(result: dict) -> QueryResponse:
    """Build a QueryResponse from a completed RAGState."""
    relevant = result.get("relevant_docs") or result.get("retrieved_docs", [])
    retrieved = result.get("retrieved_docs", [])
    grades = result.get("doc_grades", [])

    graded = [
        DocGrade(
            source=doc.metadata.get("source", "unknown"),
            grade=grade,
        )
        for doc, grade in zip(retrieved, grades)
    ] if grades else []

    sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant})

    return QueryResponse(
        answer=result["answer"],
        rewritten_query=result.get("rewritten_query", result["question"]),
        sources=sources,
        doc_grades=graded,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Liveness probe — used by Docker/k8s health checks."""
    return {"status": "ok"}


@router.get("/pipeline-info")
def pipeline_info():
    """Describe the current graph architecture for documentation/demo purposes."""
    return {
        "graph": "rewrite -> retrieve -> grade_docs -> (conditional) -> generate",
        "nodes": {
            "rewrite": "Expands user question into statutory language (query rewriting)",
            "retrieve": "Hybrid BM25 (40%) + dense (60%) search via EnsembleRetriever",
            "grade_docs": "LLM grades each chunk yes/no for relevance (CRAG)",
            "transform_query": "Triggered when all docs fail grading; generates a new query",
            "generate": "Grounded answer generation from graded-relevant chunks only",
        },
        "conditional_edge": "grade_docs -> generate if relevant docs exist, else -> transform_query",
        "max_retries": 1,
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Synchronous Q&A endpoint.

    Runs the full CRAG pipeline and returns the answer along with a pipeline
    trace showing rewritten_query, doc_grades, and which sources were used.
    """
    result = _run_graph(request.question)
    return _extract_response(result)


@router.post("/stream")
async def stream(request: QueryRequest):
    """
    Streaming Q&A endpoint using Server-Sent Events (SSE).

    Why streaming?
    --------------
    The CRAG pipeline can take 20-60s (rewrite + grade 4 docs + generate).
    Without streaming, the user stares at a blank page. With SSE, we emit
    progress events as each pipeline stage completes, then stream the final
    answer word-by-word for a typewriter effect.

    SSE format: data: <json>\\n\\n

    Event types:
      pipeline_start  — query received
      stage           — a graph node started
      grade_result    — individual doc grade result
      answer_chunk    — a word of the final answer
      done            — pipeline complete with full metadata
      error           — something went wrong
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    async def event_stream() -> AsyncGenerator[str, None]:
        def emit(event_type: str, data: dict) -> str:
            payload = json.dumps({"event": event_type, **data})
            return f"data: {payload}\n\n"

        yield emit("pipeline_start", {"question": request.question})

        try:
            yield emit("stage", {"node": "rewrite", "message": "Expanding query..."})
            result = _run_graph(request.question)

            rewritten = result.get("rewritten_query", request.question)
            yield emit("stage", {"node": "retrieve", "message": "Retrieved chunks"})
            yield emit("stage", {"node": "grade_docs", "message": "Grading chunks..."})

            retrieved = result.get("retrieved_docs", [])
            grades = result.get("doc_grades", [])
            for doc, grade in zip(retrieved, grades):
                yield emit("grade_result", {
                    "source": doc.metadata.get("source", "unknown"),
                    "grade": grade,
                })

            relevant_count = len(result.get("relevant_docs") or [])
            if relevant_count == 0:
                yield emit("stage", {
                    "node": "transform_query",
                    "message": "No relevant docs, retrying with new query...",
                })

            yield emit("stage", {"node": "generate", "message": "Generating answer..."})

            # Stream answer word-by-word for a typewriter effect
            answer = result.get("answer", "")
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield emit("answer_chunk", {"text": chunk})

            response = _extract_response(result)
            yield emit("done", {
                "rewritten_query": rewritten,
                "sources": response.sources,
                "doc_grades": [g.model_dump() for g in response.doc_grades],
            })

        except Exception as exc:
            yield emit("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Trigger a (re)index of everything in data/source_docs.

    In production: put this behind auth or move to a background job.
    Exposed here for demo convenience.
    """
    try:
        count = run_ingestion()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(chunks_indexed=count)
