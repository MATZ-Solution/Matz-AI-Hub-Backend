"""
src/nodes/safety.py
--------------------
Flags urgent/emergency queries so generate_answer can short-circuit
to a fixed "contact the relevant team immediately" response.
This avoids letting the LLM handle sensitive incidents unpredictably.
"""

from agent.src.models.state import AgentState

URGENT_KEYWORDS = [
    # IT & Security incidents
    "data breach", "hacked", "hack", "ransomware", "malware",
    "unauthorized access", "credentials leaked", "password stolen",
    "phishing", "suspicious email", "suspicious link",
    "server is down", "server down", "system down", "system is down",
    "production down", "production is down",
    "critical bug", "outage",

    # HR urgent
    "harassment", "discrimination", "misconduct", "fired", "terminated",

    # Legal / compliance
    "lawsuit", "legal action", "regulatory breach", "audit failure",
]


def safety_check(state: AgentState) -> AgentState:
    """Set is_emergency=True if the query matches any urgent keyword."""
    query = state["user_query"].lower()
    is_urgent = any(kw in query for kw in URGENT_KEYWORDS)
    return {**state, "is_emergency": is_urgent}
