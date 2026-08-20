"""
tests/test_agent.py
--------------------
Basic smoke tests for the agent graph.

Run:
    cd langgraph_agent
    pytest tests/
"""

from unittest.mock import patch


def test_chat_returns_tuple():
    """chat() should return (answer_str, history_list)."""
    with patch("src.nodes.retrieve.retrieve_context", return_value={
        "user_query": "hello",
        "chat_history": [],
        "retrieved_docs": [],
        "is_emergency": False,
        "answer": None,
    }), patch("src.nodes.generate.call_llm", return_value="Hello! How can I help?"):
        from src.agent.graph import chat
        answer, history = chat("hello")
        assert isinstance(answer, str)
        assert isinstance(history, list)


def test_emergency_short_circuits():
    """Urgent queries should return EMERGENCY_RESPONSE without calling the LLM."""
    from src.nodes.safety import safety_check
    state = {
        "user_query": "we have a data breach",
        "chat_history": [],
        "retrieved_docs": [],
        "is_emergency": False,
        "answer": None,
    }
    result = safety_check(state)
    assert result["is_emergency"] is True
