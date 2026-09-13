"""
Document ingestion pipeline: load -> chunk -> embed -> index.

Run via scripts/ingest.py. This is the offline half of the architecture
diagram — it populates the vector store the query path reads from.

Chunking strategy (for the Income Tax Act vertical):
  1. MarkdownHeaderTextSplitter splits on # / ## / ### headers, keeping
     section boundaries intact and propagating header hierarchy as metadata.
  2. Any chunk that exceeds max_chunk_size gets a secondary pass with
     RecursiveCharacterTextSplitter so nothing overflows the embedding
     model's context window.
  3. Every chunk is enriched with structured metadata (section number,
     cross-references, document type) via app.ingestion.metadata.
"""
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from app.config import settings
from app.vectorstore import get_vectorstore
from app.ingestion.metadata import enrich_chunk_metadata

LOADERS_BY_EXT = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

# Headers the markdown splitter will recognise as split-points.
# Each tuple is (markdown prefix, metadata key).
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),   # Chapter level
    ("##", "header_2"),  # Part level
    ("###", "header_3"), # Section level
]


# ---------------------------------------------------------------------------
# 1. LOAD — recursively walk source_docs/ and load every recognised file
# ---------------------------------------------------------------------------

def load_documents(source_dir: str = None) -> list[Document]:
    """Recursively load all .md / .txt / .pdf files under source_dir.

    Each Document's metadata will already contain 'source' set to the
    absolute file path, which we later replace with a relative path in
    the enrichment step.
    """
    source_dir = source_dir or settings.source_docs_dir
    docs: list[Document] = []

    for root, _dirs, files in os.walk(source_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            loader_cls = LOADERS_BY_EXT.get(ext)
            if not loader_cls:
                continue

            path = os.path.join(root, fname)
            try:
                loaded = loader_cls(path).load()
                # Tag every doc with the filepath for later metadata use
                for doc in loaded:
                    doc.metadata["_filepath"] = path
                docs.extend(loaded)
            except Exception as e:
                print(f"  ⚠  Skipping {path}: {e}")

    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK — two-pass splitting (markdown headers, then size fallback)
# ---------------------------------------------------------------------------

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents using a two-pass strategy.

    Pass 1: MarkdownHeaderTextSplitter — splits on #/##/### boundaries.
            This keeps each legal section as an atomic chunk and carries
            the header hierarchy (chapter, part, section) into metadata.

    Pass 2: RecursiveCharacterTextSplitter — a safety net for any chunk
            that exceeds max_chunk_size after Pass 1 (rare but possible
            for very long sections).
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # Keep headers in the text so the LLM sees them
    )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        filepath = doc.metadata.get("_filepath", "unknown")

        # Pass 1: split by markdown headers
        md_chunks = md_splitter.split_text(doc.page_content)

        for md_chunk in md_chunks:
            # md_chunk is a Document with page_content + metadata from headers
            # Merge header metadata from the splitter with the original metadata
            merged_meta = {**doc.metadata, **md_chunk.metadata}
            chunk = Document(page_content=md_chunk.page_content, metadata=merged_meta)

            # Pass 2: if still too big, sub-split
            if len(chunk.page_content) > settings.max_chunk_size:
                sub_chunks = fallback_splitter.split_documents([chunk])
                for sc in sub_chunks:
                    sc.metadata["was_sub_split"] = True
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# 3. ENRICH — attach structured metadata to every chunk
# ---------------------------------------------------------------------------

def enrich_chunks(chunks: list[Document]) -> list[Document]:
    """Apply metadata enrichment to all chunks.

    Adds: source (relative path), document_type, section_number,
    cross_references.
    """
    for chunk in chunks:
        filepath = chunk.metadata.get("_filepath", "unknown")
        enrich_chunk_metadata(chunk, filepath)
        # Clean up internal-only key
        chunk.metadata.pop("_filepath", None)

    return chunks


# ---------------------------------------------------------------------------
# 4. INDEX — store in the vector store
# ---------------------------------------------------------------------------

def run_ingestion(source_dir: str = None) -> int:
    """Load, chunk, enrich, and index all documents.

    Returns the number of chunks indexed.
    """
    print("📂 Loading documents...")
    documents = load_documents(source_dir)
    if not documents:
        raise RuntimeError(
            f"No documents found in {source_dir or settings.source_docs_dir}. "
            "Add .pdf/.txt/.md files first."
        )
    print(f"   Loaded {len(documents)} document(s).")

    print("✂️  Chunking...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunk(s).")

    print("🏷️  Enriching metadata...")
    chunks = enrich_chunks(chunks)

    print("📥 Indexing into vector store...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    # Print a summary of what was indexed
    print("\n--- Ingestion Summary ---")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        preview = chunk.page_content[:80].replace("\n", " ")
        print(
            f"  [{i+1:2d}] type={meta.get('document_type', '?'):12s} "
            f"section={meta.get('section_number', '-'):6s} "
            f"xrefs={meta.get('cross_references', ''):30s} "
            f"| {preview}..."
        )

    return len(chunks)
