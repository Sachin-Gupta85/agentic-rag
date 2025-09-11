# Agentic RAG Patterns

Agentic RAG adds decision-making around the basic retrieve-then-generate loop. Rather
than always retrieving once and answering, an agent reasons about *whether* and *how* to
retrieve, and can correct itself.

## Query routing

A router decides if a question needs the knowledge base at all. Greetings, small talk,
or clearly out-of-scope requests can be answered directly, saving a retrieval round trip.

## Document grading (self-reflection)

After retrieval, the agent judges whether the retrieved passages are actually relevant to
the question. This "self-reflection" step catches cases where the vector search returned
superficially similar but unhelpful text.

## Query rewriting (self-correction)

If the retrieved documents are graded irrelevant, the agent rewrites the query — making it
more explicit or keyword-rich — and retrieves again. A rewrite budget prevents infinite
loops; after the budget is exhausted, the agent returns an honest "not found" response
instead of guessing.

## Why it matters

These patterns raise answer quality on hard or ambiguous questions and reduce confident
wrong answers. The cost is extra LLM calls, so budgets and routing keep latency in check.
