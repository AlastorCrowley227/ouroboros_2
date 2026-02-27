"""Knowledge search tool via local Ollama model (no external LLM APIs)."""

from __future__ import annotations

import json
import os
from typing import List

import requests

from ouroboros.tools.registry import ToolContext, ToolEntry


def _web_search(ctx: ToolContext, query: str) -> str:
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", os.environ.get("OUROBOROS_MODEL_LIGHT", "qwen2.5:3b"))
    prompt = (
        "Ты помощник поиска. У тебя нет доступа к интернету в реальном времени. "
        "Дай лучший ответ по внутренним знаниям и явно отметь, что ответ может быть устаревшим.\n\n"
        f"Запрос: {query}"
    )
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        resp.raise_for_status()
        d = resp.json()
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        return json.dumps({"answer": text or "(no answer)", "sources": []}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": "Answer search-like queries using a local Ollama model without external APIs.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]
