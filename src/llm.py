"""Shared LLM factory (OpenAI-compatible)."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import settings


def chat(temperature: float = 0.0) -> ChatOpenAI:
    settings.validate_llm()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
    )
