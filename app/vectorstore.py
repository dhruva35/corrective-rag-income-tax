"""
Vector store abstraction.

The rest of the app only ever calls get_vectorstore() — swap Chroma for
Pinecone/Weaviate here and nothing else in the codebase needs to change.

Embedding provider is configurable via EMBEDDING_PROVIDER in .env:
  - "google"  → Google Gemini (free via AI Studio)
  - "openai"  → OpenAI text-embedding-3-small (paid)
"""
from langchain_community.vectorstores import Chroma

from app.config import settings


def _build_embeddings():
    """Create the embedding model based on settings.embedding_provider."""
    provider = settings.embedding_provider.lower()

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {provider}. Use 'google' or 'openai'."
    )


_embeddings = _build_embeddings()


def get_vectorstore():
    if settings.vector_store == "chroma":
        return Chroma(
            collection_name="rag_assistant",
            embedding_function=_embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    if settings.vector_store == "pinecone":
        # pip install langchain-pinecone pinecone-client
        from langchain_pinecone import PineconeVectorStore

        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=_embeddings,
            pinecone_api_key=settings.pinecone_api_key,
        )

    raise ValueError(f"Unknown VECTOR_STORE: {settings.vector_store}")


def get_embeddings():
    return _embeddings
