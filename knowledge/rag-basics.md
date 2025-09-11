# Retrieval-Augmented Generation (RAG) — Basics

Retrieval-Augmented Generation (RAG) is a technique that grounds a language model's
answers in an external knowledge source. Instead of relying only on parameters learned
during training, a RAG system retrieves relevant passages at query time and feeds them
to the model as context.

## Core pipeline

1. **Ingestion** — documents are split into chunks and converted into vector embeddings,
   then stored in a vector database.
2. **Retrieval** — the user's question is embedded and the most similar chunks are
   fetched (typically top-k nearest neighbours).
3. **Generation** — the retrieved chunks are inserted into the prompt, and the model
   answers using that context.

## Why use RAG

- **Freshness** — knowledge can be updated by changing documents, with no retraining.
- **Attribution** — answers can cite their sources, improving trust.
- **Reduced hallucination** — grounding the model in real text lowers fabrication,
  especially when the prompt instructs the model to answer only from context.

## Chunking

Chunk size trades off recall and precision. Larger chunks keep more context together but
dilute relevance; smaller chunks are more precise but may lose surrounding meaning. A
common starting point is 800–1000 characters with 10–15% overlap.
