import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-assistant")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "google")
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")
    embedding_model: str = "text-embedding-3-small"
    google_embedding_model: str = "models/gemini-embedding-001"
    chat_model: str = "gpt-4o-mini"
    google_chat_model: str = "gemini-flash-lite-latest"

    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunk_size: int = 1200
    retrieval_k: int = 4
    retriever_type: str = os.getenv("RETRIEVER_TYPE", "hybrid")  # "dense" or "hybrid"
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))  # 0.0=pure dense, 1.0=pure BM25
    enable_query_rewriting: bool = os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true"

    source_docs_dir: str = "data/source_docs"


settings = Settings()
