from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str               # Original user question, never modified
    rewritten_query: str        # Expanded/rewritten query used for retrieval
    retrieved_docs: List[Document]
    doc_grades: List[str]       # "yes"/"no" relevance grade per retrieved doc
    relevant_docs: List[Document]  # Only the docs that passed the grader
    retry_count: int            # Guards against infinite re-retrieval loops
    answer: str
