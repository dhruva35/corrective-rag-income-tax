"""
Hybrid retrieval: BM25 (keyword) + Dense (semantic) via EnsembleRetriever.

Why hybrid?
-----------
Pure dense/semantic search ranks chunks by embedding cosine similarity.
It's great at capturing meaning, but weak at exact-match legal terms like
"Section 123", "Schedule XV", or "Section 2(42A)". A user asking about
Section 2(42A) might retrieve chunks about Section 123 because they're
both in "Chapter I: Preliminary" — same semantic neighbourhood, wrong answer.

BM25 is a classic keyword scoring algorithm (TF-IDF variant). It ranks
chunks where the query terms appear frequently and rarely in the corpus
(high TF, high IDF). For legal/tax text, BM25 and dense are highly
complementary:

  BM25  → precise section/schedule numbers, exact statutory phrases
  Dense → paraphrased questions, concept-level retrieval

LangChain's EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge
the two ranked lists. Each chunk gets score: sum(1 / (rank_i + 60)) across
retrievers, weighted by the retriever's weight.

Architecture diagram:
  Question
    ├──► BM25Retriever   (top-k, weight=bm25_weight)  ─┐
    └──► Dense Retriever (top-k, weight=1-bm25_weight) ─┤
                                                         ▼
                                                EnsembleRetriever (RRF)
                                                         │
                                                   top-k fused docs
"""
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore import get_vectorstore


def _load_all_docs_from_chroma() -> list[Document]:
    """Pull every stored document out of Chroma to build the BM25 index.

    Chroma's low-level collection.get() returns raw data; we reconstruct
    LangChain Documents so BM25Retriever can tokenise the page_content.
    """
    vectorstore = get_vectorstore()
    # Access the underlying Chroma collection directly
    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return docs


@lru_cache(maxsize=1)
def _build_bm25_retriever() -> BM25Retriever:
    """Build and cache the BM25 index (expensive to rebuild each call).

    BM25 tokenises every stored chunk and builds an inverted index in memory.
    We cache it so the graph doesn't rebuild it on every query.
    """
    docs = _load_all_docs_from_chroma()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = settings.retrieval_k
    return retriever


def get_hybrid_retriever() -> EnsembleRetriever:
    """Return an EnsembleRetriever combining BM25 + dense search.

    Weights are configurable via settings.bm25_weight (default 0.4).
    The remaining weight (0.6) goes to the dense retriever.

    To experiment:
      - Increase bm25_weight → more keyword-driven (good for exact section refs)
      - Decrease bm25_weight → more semantic (good for paraphrased questions)
    """
    vectorstore = get_vectorstore()
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_k}
    )

    bm25_retriever = _build_bm25_retriever()

    bm25_weight = settings.bm25_weight
    dense_weight = round(1.0 - bm25_weight, 2)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
    )
