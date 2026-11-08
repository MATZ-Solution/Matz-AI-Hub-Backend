"""
tests/test_agent.py — smoke tests for the agent graph (no network).

Run from Backend/:  .venv/bin/pytest tests/test_agent.py -q
"""

from agents.langgraph_agent.nodes import node
from agents.langgraph_agent.tools import tools


def test_chat_returns_answer_citations_history_and_grounding(monkeypatch):
    monkeypatch.setattr(node, "call_llm", lambda **kw: "Hello! How can I help?")
    monkeypatch.setattr(tools, "call_llm", lambda **kw: '{"grounded": true}')
    from agents.langgraph_agent.main_langgraph_agent import chat

    answer, citations, history, grounded = chat("hello", [], organization_id="org-a")
    assert isinstance(answer, str) and answer
    assert citations == []
    assert isinstance(history, list)
    assert isinstance(grounded, bool)


def test_emergency_short_circuits():
    """Urgent queries are flagged without calling the LLM."""
    state = {
        "user_query": "we have a data breach",
        "chat_history": [],
        "retrieved_docs": [],
        "is_emergency": False,
        "answer": None,
    }
    assert node.safety_check(state)["is_emergency"] is True
