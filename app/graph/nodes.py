import time
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.vectorstore import get_vectorstore
from app.graph.state import RAGState

def _build_llm():
    if settings.llm_provider.lower() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.google_chat_model,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

_llm = _build_llm()

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the context doesn't contain the answer, say "
            "you don't know — never make something up. Cite which source each "
            "fact came from using the [source] tags in the context.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at reformulating questions for a legal/tax document "
            "retrieval system. Your goal is to rewrite the user's question so it:\n"
            "1. Spells out abbreviations (e.g. 80C → Section 123 of the Income Tax Act 2025, "
            "NPS → National Pension System)\n"
            "2. Includes the exact statutory terms a legal document would use\n"
            "3. If the question has multiple parts, separates them clearly\n\n"
            "Return ONLY the rewritten question(s). No explanation, no preamble.",
        ),
        (
            "human",
            "Original question: {question}",
        ),
    ]
)

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader for a legal document retrieval system.\n"
            "Given a document chunk and a user question, assess whether the chunk \n"
            "contains information that would help answer the question.\n\n"
            "Be generous: even partial or background context counts as relevant.\n"
            "Return ONLY the word 'yes' or 'no' — nothing else.",
        ),
        (
            "human",
            "Question: {question}\n\nDocument chunk:\n{document}",
        ),
    ]
)

_TRANSFORM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at improving search queries for a legal/tax retrieval system.\n"
            "The previous search returned no relevant results. Generate a completely \n"
            "different, more targeted search query that:\n"
            "1. Uses different keywords or phrasing than the original\n"
            "2. Focuses on the core legal concept being asked about\n"
            "3. Is concise (one sentence)\n\n"
            "Return ONLY the new search query. No explanation.",
        ),
        (
            "human",
            "Original question: {question}\nFirst attempt query: {rewritten_query}",
        ),
    ]
)


def rewrite_node(state: RAGState) -> RAGState:
    """Expand and clarify the question before retrieval.

    Why query rewriting?
    --------------------
    Users ask questions in natural language: "Can I claim both 80C and NPS?"
    Legal documents use statutory language: "Section 123", "Section 124(1B)",
    "National Pension System", "Schedule XV".

    The gap between user language and document language hurts both retrievers:
    - BM25 misses because the query tokens don't appear in the document
    - Dense misses because the embedding space doesn't perfectly bridge the gap

    The rewriter bridges this gap by:
    1. Expanding abbreviations to their statutory equivalents
    2. Splitting multi-part questions so each part retrieves its own chunks
    3. Adding domain-specific terminology the user omitted

    This node is toggled by settings.enable_query_rewriting. When disabled,
    it sets rewritten_query = question (pass-through) so the rest of the
    graph is unaffected.
    """
    if not settings.enable_query_rewriting:
        # Pass-through: retrieval will use the original question
        return {**state, "rewritten_query": state["question"]}

    chain = _REWRITE_PROMPT | _llm
    response = chain.invoke({"question": state["question"]})
    rewritten = response.content.strip()

    # Log so you can see what the rewriter is doing
    print(f"  [rewrite] original : {state['question'][:80]}")
    print(f"  [rewrite] rewritten: {rewritten[:160]}")

    return {**state, "rewritten_query": rewritten}


def retrieve_node(state: RAGState) -> RAGState:
    """Pull the top-k most relevant chunks.

    Reads from `rewritten_query` (set by rewrite_node) so retrieval always
    benefits from any query expansion. Falls back to the original question
    if rewritten_query is empty (e.g. graph invoked without the rewrite node).

    Retriever is configurable via settings.retriever_type:
      "hybrid" (default) → BM25 (40%) + Dense (60%) via EnsembleRetriever
      "dense"             → Pure semantic similarity search (baseline)
    """
    query = state.get("rewritten_query") or state["question"]

    if settings.retriever_type == "hybrid":
        from app.retrieval.hybrid import get_hybrid_retriever
        retriever = get_hybrid_retriever()
        docs = retriever.invoke(query)[:settings.retrieval_k]
    else:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=settings.retrieval_k)

    return {**state, "retrieved_docs": docs}


def grade_docs_node(state: RAGState) -> RAGState:
    """Grade each retrieved doc for relevance — the core of Corrective RAG (CRAG).

    Why relevance grading?
    ----------------------
    Not every retrieved chunk is actually useful. The retriever returns the
    *most similar* chunks — but "most similar" doesn't mean "actually answers
    the question". For example, when asked about capital gains tax rates, the
    retriever might return a chunk about deductions because it shares many
    keywords (income tax, section, assessee).

    The grader uses the LLM to make a binary yes/no judgment on each chunk:
      - yes → keep it, pass it to generate
      - no  → discard it

    If ALL chunks are graded 'no', the graph routes to transform_query_node
    instead of generate_node. This is the 'corrective' step.

    Note: We grade against the original question (not the rewritten one) so
    the grader uses the same intent the user expressed.
    """
    question = state["question"]
    docs = state["retrieved_docs"]
    grades = []
    relevant = []

    chain = _GRADE_PROMPT | _llm

    for doc in docs:
        # Throttle to stay within free-tier 15 RPM limit.
        # Each grading call + rewrite + transform + generate can exceed
        # the limit in rapid succession without this pause.
        time.sleep(4)
        # Truncate chunk to avoid excessive tokens in the grading call
        snippet = doc.page_content[:800]
        response = chain.invoke({"question": question, "document": snippet})
        grade = response.content.strip().lower()
        # Normalise: treat anything starting with 'y' as yes
        grade = "yes" if grade.startswith("y") else "no"
        grades.append(grade)
        if grade == "yes":
            relevant.append(doc)

    print(f"  [grade] {len(relevant)}/{len(docs)} docs passed relevance check")
    for doc, g in zip(docs, grades):
        src = doc.metadata.get("source", "?")
        print(f"    [{g}] {src}")

    return {**state, "doc_grades": grades, "relevant_docs": relevant}


def transform_query_node(state: RAGState) -> RAGState:
    """Generate a fresh search query when the first retrieval found nothing useful.

    This is called when grade_docs_node finds zero relevant chunks.
    Rather than re-using the same (failed) query, we ask the LLM to think of
    a completely different angle on the same question.

    The result is stored back in `rewritten_query` so retrieve_node picks it
    up on the next pass. retry_count is incremented to prevent infinite loops
    (the graph will stop re-trying after 1 fallback attempt).
    """
    chain = _TRANSFORM_PROMPT | _llm
    response = chain.invoke({
        "question": state["question"],
        "rewritten_query": state.get("rewritten_query", state["question"]),
    })
    new_query = response.content.strip()
    retry_count = state.get("retry_count", 0) + 1

    print(f"  [transform] retry #{retry_count}: {new_query[:120]}")

    return {**state, "rewritten_query": new_query, "retry_count": retry_count}


def decide_to_generate(state: RAGState) -> str:
    """Routing function for the conditional edge after grade_docs.

    Returns a string that LangGraph uses to decide the next node:
      'generate'         → at least one chunk was relevant, proceed normally
      'transform_query'  → no relevant chunks AND we haven't retried yet
      'generate'         → no relevant chunks BUT retry_count >= 1 (give up,
                           generate with empty/best-effort context rather
                           than loop forever)

    This is a pure function — it only reads state, never writes it.
    LangGraph calls it after grade_docs_node finishes.
    """
    relevant = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if relevant:
        print(f"  [route] \u2713 {len(relevant)} relevant docs found \u2192 generate")
        return "generate"
    if retry_count < 1:
        print("  [route] \u2717 no relevant docs, retry_count=0 \u2192 transform_query")
        return "transform_query"
    print("  [route] \u2717 no relevant docs after retry \u2192 generate (best-effort)")
    return "generate"


def generate_node(state: RAGState) -> RAGState:
    """Generate a grounded answer from the graded-relevant chunks.

    Uses `relevant_docs` (set by grade_docs_node) if available.
    Falls back to `retrieved_docs` if relevance grading was skipped
    (e.g. when running the graph without the grade node for testing).
    """
    # Prefer the grader-filtered list; fall back gracefully
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])

    if not docs:
        return {**state, "answer": "I could not find relevant information in the tax documents to answer this question."}

    context = "\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
    chain = _PROMPT | _llm
    response = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": response.content}
