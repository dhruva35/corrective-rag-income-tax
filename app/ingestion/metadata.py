"""
Metadata extraction utilities for Income Tax Act documents.

Parses section numbers, cross-references, and document types from chunks
so the vector store can support filtered and multi-hop retrieval.
"""
import re
import os
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Section number extraction
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"(?:^|\n)\s*#{1,4}\s*Section\s+(\d+[A-Z]*)\b",
    re.IGNORECASE,
)


def extract_section_number(text: str) -> str | None:
    """Pull the primary section number from a chunk's text.

    Looks for markdown headings like '### Section 123.' and returns '123'.
    Returns None if no section heading is found (e.g. for schedule or
    concordance chunks).
    """
    match = _SECTION_RE.search(text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Cross-reference extraction
# ---------------------------------------------------------------------------

_XREF_SECTION_RE = re.compile(r"[Ss]ection\s+(\d+[A-Z]*(?:\(\d+[A-Z]*\))*)")
_XREF_SCHEDULE_RE = re.compile(r"[Ss]chedule\s+([IVXLCDM]+\b|[A-Z]+\b)")


def extract_cross_references(text: str, own_section: str | None = None) -> list[str]:
    """Find all Section and Schedule references in a chunk.

    Returns deduplicated, sorted list like ['Section 124', 'Schedule XV'].
    Excludes the chunk's own section number to avoid self-references.
    """
    refs: set[str] = set()

    for m in _XREF_SECTION_RE.finditer(text):
        sec = m.group(1)
        # Skip self-reference
        if own_section and sec.startswith(own_section):
            continue
        refs.add(f"Section {sec}")

    for m in _XREF_SCHEDULE_RE.finditer(text):
        refs.add(f"Schedule {m.group(1)}")

    return sorted(refs)


# ---------------------------------------------------------------------------
# Document type inference
# ---------------------------------------------------------------------------

_DIR_TO_DOCTYPE = {
    "act_2025": "bare_act",
    "schedules": "schedule",
    "concordance": "concordance",
    "circulars": "circular",
}


def infer_document_type(filepath: str) -> str:
    """Map the subdirectory name to a document type label.

    e.g. 'data/source_docs/act_2025/ch08_deductions.md' → 'bare_act'
    Falls back to 'other' for unrecognized directories.
    """
    # Normalize path separators
    parts = filepath.replace("\\", "/").split("/")

    # Walk backwards looking for a known directory name
    for part in reversed(parts):
        if part in _DIR_TO_DOCTYPE:
            return _DIR_TO_DOCTYPE[part]

    return "other"


# ---------------------------------------------------------------------------
# Chunk enrichment (ties everything together)
# ---------------------------------------------------------------------------


def enrich_chunk_metadata(chunk: Document, filepath: str) -> Document:
    """Attach structured metadata to a LangChain Document chunk.

    Adds: source, document_type, section_number, cross_references.
    """
    text = chunk.page_content

    # Source path relative to source_docs/
    source_marker = "source_docs"
    norm = filepath.replace("\\", "/")
    idx = norm.find(source_marker)
    relative = norm[idx + len(source_marker) + 1:] if idx != -1 else norm

    section = extract_section_number(text)
    xrefs = extract_cross_references(text, own_section=section)

    chunk.metadata.update({
        "source": relative,
        "document_type": infer_document_type(filepath),
        "section_number": section or "",
        "cross_references": ", ".join(xrefs) if xrefs else "",
    })

    return chunk
