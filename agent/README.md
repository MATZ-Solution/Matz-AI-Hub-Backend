# MATZ Health Chatbot — LangGraph Agent

An AI knowledge assistant that answers employee questions from company documents (HR policies, security guidelines, SOPs, etc.) using RAG (Retrieval-Augmented Generation).

---

## Project Structure

```
langgraph_agent/
├── src/
│   ├── agent/          # Graph wiring — builds & compiles the LangGraph
│   │   └── graph.py
│   ├── models/         # State schema + LLM client
│   │   ├── state.py
│   │   └── llm_client.py
│   ├── nodes/          # One file per graph node
│   │   ├── retrieve.py   # Qdrant vector search
│   │   ├── safety.py     # Emergency keyword detection
│   │   └── generate.py   # Groq LLM response generation
│   ├── prompts/        # System and response prompt strings
│   │   └── system_prompt.py
│   └── utils/          # Shared helpers
│       ├── config.py     # All env vars in one place
│       └── logger.py     # Centralised logger
├── data/
│   └── knowledge_base/
│       └── upload_to_qdrant.py   # One-time KB upload script
├── tests/
│   └── test_agent.py
├── logs/
├── .env                # API keys (never commit this)
├── .gitignore
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd langgraph_agent
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` (or create `.env` manually):

```env
GROQ_API_KEY=your-groq-key-here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key-here
```

- Groq key: https://console.groq.com/keys (free)
- Qdrant cloud: https://cloud.qdrant.io (free tier available)

### 3. Upload the knowledge base (first time only)

```bash
python data/knowledge_base/upload_to_qdrant.py
```

### 4. Run the chatbot

```bash
python main.py
```

---

## How It Works

```
User query
    │
    ▼
[retrieve_context]  →  Embeds query, searches Qdrant with tenant filter
    │
    ▼
[safety_check]      →  Detects urgent/emergency keywords
    │
    ▼
[generate_answer]   →  Calls Groq LLM with context + history
    │
    ▼
Answer + updated chat history
```

---

## Running Tests

```bash
pytest tests/
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Lazy singletons for Qdrant client & embed model | Avoid cold-start latency on every request |
| Emergency short-circuit before LLM | Reliable, deterministic response for critical incidents |
| Context injected only on current turn | History stays clean; no context bleed across turns |
| All config in `src/utils/config.py` | Single source of truth for env vars |
