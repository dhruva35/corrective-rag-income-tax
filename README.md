# corrective-rag-income-tax

> A production-style **Corrective RAG (CRAG)** pipeline over the Indian Income Tax Act 2025, built with LangChain, LangGraph, and FastAPI. Features hybrid retrieval, LLM-based relevance grading, conditional re-retrieval, and a streaming Chat UI.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=chainlink)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-DAG-purple)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://trychroma.com)

---

> [!NOTE]
> **Live Demo Limitation:** The deployed UI is a Proof of Concept built over a very small, specific corpus (only 4 markdown files containing specific tax provisions). If you ask general questions (e.g., "What is income tax?"), the CRAG grader will correctly reject the specific retrieved chunks and respond with *"I don't know"*. This is the anti-hallucination feature working exactly as designed! To answer general questions, you would simply ingest a broader set of source documents.

## What This Is

A **RAG (Retrieval-Augmented Generation)** system that answers questions about the Indian Income Tax Act 2025. Unlike a standard RAG pipeline, this implements **CRAG (Corrective RAG)** — a self-correcting architecture that:

1. **Rewrites** the user's question into precise statutory language before searching
2. **Retrieves** relevant document chunks using Hybrid Search (BM25 + semantic vectors)
3. **Grades** each chunk for relevance using an LLM judge (yes/no per chunk)
4. **Conditionally re-retrieves** with a transformed query if all chunks fail grading
5. **Generates** a grounded answer using only the verified-relevant chunks
6. Returns **"I don't know"** instead of hallucinating when no relevant context is found

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE — One-time Ingestion                                    │
│                                                                 │
│  Markdown Docs → RecursiveTextSplitter → Metadata Tagging       │
│               → Google text-embedding-004 → ChromaDB            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RUNTIME — Per Query (LangGraph DAG)                            │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  [rewrite_node]  ← Expands abbreviations, adds statutory terms  │
│      │                                                          │
│      ▼                                                          │
│  [retrieve_node] ← BM25 (40%) + Dense (60%) → RRF fusion       │
│      │                                                          │
│      ▼                                                          │
│  [grade_docs_node] ← LLM judges each chunk: YES / NO           │
│      │                                                          │
│      ├─── relevant docs found ──────────► [generate_node] ──► Answer
│      │                                                          │
│      └─── 0 relevant docs ──► [transform_query_node]           │
│                                      │                          │
│                                      └──► [retrieve_node] ──►  │
│                                           [grade_docs_node] ──► │
│                                           [generate_node] ──► Answer
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Progress (8 Steps)

| Step | What Was Built | Key Concept |
|---|---|---|
| 1 | Domain selection & project scaffold | Vertical RAG strategy |
| 2 | Markdown ingestion → ChromaDB | Chunking, metadata tagging, embeddings |
| 3 | Baseline evaluation | RAGAS-style metrics (faithfulness, precision, recall) |
| 4 | Hybrid Retrieval (BM25 + Dense + RRF) | `EnsembleRetriever`, Reciprocal Rank Fusion |
| 5 | Query Rewriting node | LangGraph multi-node DAG, state |
| 6 | CRAG: Relevance Grading + Conditional Re-retrieval | `add_conditional_edges()`, graph loops |
| 7 | FastAPI REST API + SSE Streaming | `StreamingResponse`, Server-Sent Events |
| 8 | Dark-mode Chat UI | SSE client, live pipeline trace panel |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini Flash Lite (`gemini-flash-lite-latest`) |
| **Embeddings** | Google `text-embedding-004` (768-dim) |
| **Vector Store** | ChromaDB (local disk) |
| **Keyword Search** | BM25 via `rank-bm25` |
| **Orchestration** | LangGraph `StateGraph` with conditional edges |
| **Framework** | LangChain |
| **API** | FastAPI + Uvicorn |
| **Streaming** | Server-Sent Events (SSE) |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/dhruva35/corrective-rag-income-tax.git
cd corrective-rag-income-tax
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Google AI Studio key:
# GOOGLE_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com)

### 3. Build the vector index

```bash
python scripts/ingest.py
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Open the Chat UI

Visit **http://localhost:8000/ui** in your browser.

Or use the API directly:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What investments qualify under Section 123?"}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/pipeline-info` | Current graph architecture as JSON |
| `POST` | `/query` | Synchronous Q&A with full pipeline trace |
| `POST` | `/stream` | SSE streaming Q&A with per-node progress events |
| `POST` | `/ingest` | Re-index source documents |

**Swagger UI:** `http://localhost:8000/docs`

### Example response from `/query`

```json
{
  "answer": "Based on Schedule XV, eligible investments include: LIC premium, PPF, EPF...",
  "rewritten_query": "What specific investments are eligible for deductions under Section 123 of the Income Tax Act 2025?",
  "sources": ["schedules/schedule_XV_deduction_list.md", "act_2025/ch08_deductions.md"],
  "doc_grades": [
    {"source": "schedules/schedule_XV_deduction_list.md", "grade": "yes"},
    {"source": "act_2025/ch08_deductions.md", "grade": "yes"},
    {"source": "concordance/section_mapping.md", "grade": "yes"},
    {"source": "act_2025/ch08_deductions.md", "grade": "no"}
  ]
}
```

---

## Docker

```bash
docker build -t corrective-rag-income-tax .
docker run -e GOOGLE_API_KEY=your_key -p 8000:8000 corrective-rag-income-tax
```

---

## Project Structure

```
corrective-rag-income-tax/
├── app/
│   ├── api/routes.py          # FastAPI endpoints (/query, /stream, /ingest)
│   ├── graph/
│   │   ├── build_graph.py     # LangGraph DAG wiring
│   │   ├── nodes.py           # All 5 nodes + decide_to_generate()
│   │   └── state.py           # RAGState TypedDict
│   ├── ingestion/             # Document loading, chunking, metadata
│   ├── retrieval/hybrid.py    # BM25 + Dense + EnsembleRetriever
│   ├── vectorstore.py         # ChromaDB interface
│   ├── config.py              # Settings (toggles, k values)
│   └── main.py                # FastAPI app + CORS + startup warm-up
├── data/source_docs/          # Markdown source documents
├── eval/                      # RAGAS-style evaluation harness
├── frontend/index.html        # Dark-mode streaming Chat UI
├── scripts/ingest.py          # One-time ingestion script
├── Dockerfile
├── render.yaml                # One-click Render.com deploy
└── requirements.txt
```

---

## Extending This Project

- **Add more documents:** Drop `.md`/`.txt`/`.pdf` files into `data/source_docs/` and re-run `python scripts/ingest.py`
- **Swap the LLM:** Change `model` in `app/config.py` — any LangChain-compatible LLM works
- **Swap the vector store:** Everything is abstracted in `app/vectorstore.py` — drop in Pinecone or Weaviate
- **Add metadata filtering:** Use `section_number` tags attached during ingestion to pre-filter Chroma before search

---

## Why CRAG?

Standard RAG always generates an answer, even when retrieved chunks are irrelevant (hallucination risk). CRAG adds a **verification step** — an LLM judge that grades each chunk before generation. If all chunks fail, the system rewrites the query and tries once more, then either answers from verified context or explicitly says *"I don't know."*

This makes CRAG significantly more reliable for high-stakes domains like legal and tax documents where hallucinations are dangerous.
