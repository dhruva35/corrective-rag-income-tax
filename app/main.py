"""
FastAPI application entry point.

Run with:
  uvicorn app.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (Swagger UI — try the endpoints live)
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.api.routes import router

app = FastAPI(
    title="RAG Assistant — Income Tax Act 2025",
    description=(
        "A Corrective RAG (CRAG) pipeline built with LangChain + LangGraph.\n\n"
        "**Pipeline:** `rewrite → retrieve → grade_docs → (conditional) → generate`\n\n"
        "- **Hybrid retrieval**: BM25 (40%) + dense embeddings (60%) via EnsembleRetriever\n"
        "- **Query rewriting**: LLM expands abbreviations into statutory language\n"
        "- **Relevance grading**: LLM scores each retrieved chunk yes/no\n"
        "- **Conditional re-retrieval**: If all chunks fail grading, transforms the query and retries once\n"
    ),
    version="0.6.0",
)

# CORS — allows the API to be called from a browser (e.g. a React demo UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production: ["https://yourdomain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the Chat UI from /ui
# Works both locally (http://localhost:8000/ui) and on Render
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend), html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to the Chat UI."""
    return RedirectResponse(url="/ui")


@app.on_event("startup")
def warm_up():
    """
    Pre-build the BM25 index at startup so the first /query call is fast.

    Without this, the first request triggers _build_bm25_retriever() which
    scans all Chroma documents — noticeable latency (~1-2s). Warming up here
    means it runs once in the background when the server starts.

    The lru_cache in hybrid.py ensures the result is reused for all requests.
    """
    try:
        from app.retrieval.hybrid import _build_bm25_retriever
        _build_bm25_retriever()
        print("[startup] BM25 index warmed up successfully")
    except Exception as exc:
        print(f"[startup] BM25 warm-up skipped: {exc}")
