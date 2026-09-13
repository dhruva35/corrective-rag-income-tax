from langgraph.graph import StateGraph, END

from app.graph.state import RAGState
from app.graph.nodes import (
    rewrite_node,
    retrieve_node,
    grade_docs_node,
    transform_query_node,
    generate_node,
    decide_to_generate,
)


def build_rag_graph():
    """
    Wires up a Corrective-RAG graph with a conditional branch:

        rewrite → retrieve → grade_docs ──► generate → END
                                  │
                                  │ (no relevant docs & retry_count < 1)
                                  ▼
                           transform_query → retrieve → grade_docs → generate

    Node responsibilities:
      rewrite         : Expand user question into statutory language (Step 5)
      retrieve        : Hybrid BM25 + dense search (Step 4)
      grade_docs      : LLM judges each chunk yes/no for relevance (Step 6)
      transform_query : If zero relevant chunks, generate a different query
                        and loop back to retrieve for one more attempt
      generate        : Produce grounded answer from relevant chunks

    The KEY concept here is add_conditional_edges():
      - It calls decide_to_generate(state) after grade_docs_node runs
      - The return value ("generate" or "transform_query") tells LangGraph
        which node to visit next
      - This is what makes the graph non-linear
    """
    graph = StateGraph(RAGState)

    # Register all nodes
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("generate", generate_node)

    # Linear edges
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade_docs")

    # THE KEY STEP: conditional edge from grade_docs
    # decide_to_generate() returns "generate" or "transform_query"
    graph.add_conditional_edges(
        "grade_docs",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )

    # transform_query loops back to retrieve (one retry)
    # After the second grade_docs pass, retry_count >= 1, so it always
    # routes to generate regardless of relevance
    graph.add_edge("transform_query", "retrieve")

    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_rag_graph()


