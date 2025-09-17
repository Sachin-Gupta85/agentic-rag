"""Agentic RAG as a LangGraph state machine.

Flow:
    route ──┐ (knowledge-base question)
            ├─▶ retrieve ─▶ grade ──(relevant)──▶ generate ─▶ END
            │                   │
            │                   └──(not relevant & budget left)──▶ rewrite ─▶ retrieve
            └─▶ direct ─▶ END   (small-talk / out-of-scope)

The agent decides whether to consult the knowledge base, judges whether the
retrieved context is actually relevant, and rewrites its own query (up to a
budget) before falling back to an honest "not found" answer.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import settings
from .llm import chat
from .store import get_store


class AgentState(TypedDict, total=False):
    question: str          # current (possibly rewritten) query
    original_question: str  # what the user actually asked
    documents: list[Document]
    generation: str
    rewrites: int
    _route: str            # "knowledge_base" | "direct"
    _relevant: bool        # grader verdict


# ---- Structured outputs for the two "judgement" steps -------------------------

class RouteDecision(BaseModel):
    destination: Literal["knowledge_base", "direct"] = Field(
        description="Route to 'knowledge_base' for questions needing the docs, "
        "'direct' for greetings/small-talk/out-of-scope."
    )


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="True if the documents can answer the question.")


# ---- Nodes --------------------------------------------------------------------

def route(state: AgentState) -> AgentState:
    router = chat().with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You route a user message. Choose 'knowledge_base' if answering "
                       "needs factual lookup in a document store, else 'direct'."),
            ("human", "{q}"),
        ]
    )
    decision: RouteDecision = (prompt | router).invoke({"q": state["question"]})
    return {"_route": decision.destination}  # type: ignore[return-value]


def retrieve(state: AgentState) -> AgentState:
    store = get_store()
    docs = store.as_retriever(search_kwargs={"k": settings.top_k}).invoke(
        state["question"]
    )
    return {"documents": docs}


def grade(state: AgentState) -> AgentState:
    grader = chat().with_structured_output(RelevanceGrade)
    context = "\n\n".join(d.page_content for d in state.get("documents", []))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Decide if the retrieved context is relevant enough to answer "
                       "the question. Be strict."),
            ("human", "Question: {q}\n\nContext:\n{c}"),
        ]
    )
    result: RelevanceGrade = (prompt | grader).invoke(
        {"q": state["original_question"], "c": context}
    )
    return {"_relevant": result.relevant}  # type: ignore[return-value]


def rewrite(state: AgentState) -> AgentState:
    rewriter = chat(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the user's question to improve document retrieval. "
                       "Keep the intent; make it more explicit and keyword-rich. "
                       "Return only the rewritten question."),
            ("human", "{q}"),
        ]
    )
    new_q = (prompt | rewriter | StrOutputParser()).invoke(
        {"q": state["original_question"]}
    )
    return {"question": new_q.strip(), "rewrites": state.get("rewrites", 0) + 1}


def generate(state: AgentState) -> AgentState:
    llm = chat()
    context = "\n\n".join(
        f"[{d.metadata.get('source', '')}]\n{d.page_content}"
        for d in state.get("documents", [])
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using ONLY the context. If it isn't there, say you "
                       "couldn't find it. Cite sources in brackets."),
            ("human", "Question: {q}\n\nContext:\n{c}"),
        ]
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"q": state["original_question"], "c": context}
    )
    return {"generation": answer}


def direct_answer(state: AgentState) -> AgentState:
    llm = chat(temperature=0.5)
    answer = (llm | StrOutputParser()).invoke(state["question"])
    return {"generation": answer}


def give_up(state: AgentState) -> AgentState:
    return {
        "generation": "I couldn't find enough relevant information in the knowledge "
        "base to answer that confidently."
    }


# ---- Conditional edges --------------------------------------------------------

def after_route(state: AgentState) -> str:
    return "retrieve" if state.get("_route") == "knowledge_base" else "direct"


def after_grade(state: AgentState) -> str:
    if state.get("_relevant"):
        return "generate"
    if state.get("rewrites", 0) < settings.max_query_rewrites:
        return "rewrite"
    return "give_up"


# ---- Graph assembly -----------------------------------------------------------

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("direct", direct_answer)
    g.add_node("give_up", give_up)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", after_route, {"retrieve": "retrieve", "direct": "direct"}
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("direct", END)
    g.add_edge("give_up", END)
    return g.compile()


def ask(question: str) -> str:
    agent = build_agent()
    final = agent.invoke(
        {"question": question, "original_question": question, "rewrites": 0}
    )
    return final.get("generation", "(no answer produced)")
