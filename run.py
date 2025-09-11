"""Interactive CLI for the agentic RAG agent.

Usage:
    python run.py                 # chat loop
    python run.py "your question" # one-shot
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import ask  # noqa: E402


def main(argv: list[str]) -> int:
    if argv:
        print(ask(" ".join(argv)))
        return 0

    print("Agentic RAG — ask a question (Ctrl-C or 'exit' to quit).\n")
    try:
        while True:
            q = input("you › ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                print(f"bot › {ask(q)}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
