# 🤖 agentic-rag

> A **self-correcting** RAG agent built with **LangGraph**: it decides whether to retrieve, grades whether the retrieved context is actually relevant, and rewrites its own query when retrieval is weak — before ever answering.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-ff9800)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

Plain RAG retrieves once and hopes the context is good. This agent adds **reasoning
around retrieval** — routing, self-reflection, and self-correction — as an explicit
LangGraph state machine.

---

## 🧭 The graph

```
                      ┌──────────┐
          question ─▶ │  route   │
                      └────┬─────┘
          direct ◀────────┤ else        │ needs knowledge base
            │              ▼
            │        ┌──────────┐    ┌──────────┐
            │        │ retrieve │─▶  │  grade   │
            │        └────▲─────┘    └────┬─────┘
            │             │ rewrite       │
            │        ┌────┴─────┐  not    │ relevant
            │        │ rewrite  │◀─rel.── │
            │        └──────────┘         ▼
            │              (budget)  ┌──────────┐
            │                        │ generate │
            ▼                        └────┬─────┘
           END ◀───────────────────────── ┘
```

- **Route** — skip retrieval for greetings/out-of-scope messages.
- **Grade** — an LLM judges if retrieved docs can actually answer the question.
- **Rewrite** — on weak retrieval, reformulate the query and try again (bounded budget).
- **Give up honestly** — after the budget, say "not found" instead of hallucinating.

## ✨ Features

- Explicit **LangGraph** state machine (easy to extend / visualise).
- **Structured-output** router and grader (Pydantic schemas, not brittle string parsing).
- Ships with a small **knowledge base** so it runs end-to-end out of the box.
- Local embeddings; any OpenAI-compatible LLM (incl. local Ollama).

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # add OPENAI_API_KEY (or Ollama block)

python run.py                # interactive chat (indexes knowledge/ on first run)
python run.py "What is query rewriting in agentic RAG?"
```

Add your own `.md` files to `knowledge/` and delete `vector_store/` to re-index.

## 🗂️ Layout

```
agentic-rag/
├── run.py                 # CLI entry point
├── src/
│   ├── graph.py           # the LangGraph agent (route → retrieve → grade → rewrite → generate)
│   ├── store.py           # knowledge/ → Chroma vector store
│   ├── llm.py             # OpenAI-compatible LLM factory
│   └── config.py
├── knowledge/             # sample markdown knowledge base
└── requirements.txt
```

## ⚙️ Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | LLM endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `TOP_K` | `4` | Passages retrieved |
| `MAX_QUERY_REWRITES` | `2` | Self-correction budget |

## 📝 License

MIT — see [LICENSE](LICENSE).
