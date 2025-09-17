"""Vector store: build from the local `knowledge/` folder, or load if it exists."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import KNOWLEDGE_DIR, VECTOR_STORE_DIR, settings


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_store() -> Chroma:
    """Index every .md/.txt file under knowledge/ into a persistent Chroma store."""
    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No .md files found in {KNOWLEDGE_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(VECTOR_STORE_DIR),
    )


def get_store() -> Chroma:
    """Load the persisted store, building it on first run."""
    if VECTOR_STORE_DIR.exists():
        return Chroma(
            collection_name=settings.collection_name,
            embedding_function=_embeddings(),
            persist_directory=str(VECTOR_STORE_DIR),
        )
    return build_store()
