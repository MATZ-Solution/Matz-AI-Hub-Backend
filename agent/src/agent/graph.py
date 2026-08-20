"""
src/agent/graph.py
-------------------
Full MATZ agent graph with overview path.

Flow:

    route_query
        ├── injection/urgent/small_talk/out_of_scope → generate_answer → hallucination → citation → END
        ├── overview  → retrieve_all → generate_answer → hallucination → citation → END
        └── knowledge → retrieve_context
                              ↓
                       grade_relevance
                              ├── needs_retry → retrieve_context (max 2x)
                              └── graded     → generate_answer → hallucination → citation → END
"""

from langgraph.graph import StateGraph, END

from agent.src.models.state import AgentState
from agent.src.nodes.tools.router import route_query
from agent.src.nodes.tools.relevance_grader import grade_relevance
from agent.src.nodes.tools.hallucination_checker import check_hallucination
from agent.src.nodes.tools.citation_builder import build_citations
from agent.src.nodes.retrieve import retrieve_context, retrieve_all
from agent.src.nodes.generate import generate_answer


def _decide_after_router(state: AgentState) -> str:
    query_type = state.get("query_type", "knowledge")
    if query_type in ("urgent", "injection", "small_talk", "out_of_scope"):
        return "generate_answer"
    elif query_type == "overview":
        return "retrieve_all"
    else:
        return "retrieve_context"


def _decide_after_grader(state: AgentState) -> str:
    if state.get("needs_retry", False):
        return "retrieve_context"
    return "generate_answer"


def _decide_after_hallucination(state: AgentState) -> str:
    is_grounded = state.get("is_grounded", True)
    attempts    = state.get("generation_attempts", 0)
    if not is_grounded and attempts < 2:
        return "generate_answer"
    return "citation_builder"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route_query",           route_query)
    graph.add_node("retrieve_context",      retrieve_context)
    graph.add_node("retrieve_all",          retrieve_all)       # ← new
    graph.add_node("grade_relevance",       grade_relevance)
    graph.add_node("generate_answer",       generate_answer)
    graph.add_node("hallucination_checker", check_hallucination)
    graph.add_node("citation_builder",      build_citations)

    graph.set_entry_point("route_query")

    graph.add_conditional_edges(
        "route_query",
        _decide_after_router,
        {
            "retrieve_context": "retrieve_context",
            "retrieve_all":     "retrieve_all",
            "generate_answer":  "generate_answer",
        }
    )

    # retrieve_all skips grading — all chunks are passed directly
    graph.add_edge("retrieve_all", "generate_answer")

    graph.add_edge("retrieve_context", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        _decide_after_grader,
        {
            "retrieve_context": "retrieve_context",
            "generate_answer":  "generate_answer",
        }
    )

    graph.add_edge("generate_answer", "hallucination_checker")

    graph.add_conditional_edges(
        "hallucination_checker",
        _decide_after_hallucination,
        {
            "generate_answer":  "generate_answer",
            "citation_builder": "citation_builder",
        }
    )

    graph.add_edge("citation_builder", END)

    return graph.compile()


compiled_graph = build_graph()


def chat(user_query: str, chat_history: list = None) -> tuple:
    if chat_history is None:
        chat_history = []

    initial_state: AgentState = {
        "user_query":          user_query,
        "chat_history":        chat_history,
        "retrieved_docs":      [],
        "is_emergency":        False,
        "answer":              None,
        "query_type":          None,
        "target_collection":   None,
        "search_strategy":     None,
        "graded_docs":         [],
        "needs_retry":         False,
        "retrieval_attempts":  0,
        "rewritten_query":     None,
        "is_grounded":         False,
        "generation_attempts": 0,
        "citations":           [],
    }
    result = compiled_graph.invoke(initial_state)
    return result["answer"], result["citations"], result["chat_history"]