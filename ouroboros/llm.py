# =============================================================================
# llm.py (финальная версия с прогревом и разделением эндпоинтов)
# =============================================================================
"""
Ouroboros — local LLM client via Ollama.

The only module that communicates with the LLM API.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "qwen2.5:3b"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_ollama_models(base_url: str) -> List[str]:
    """Return installed Ollama model names."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                out.append(name)
        return out
    except Exception:
        log.debug("Failed to fetch Ollama model list", exc_info=True)
        return []


class LLMClient:
    """Ollama API wrapper. All LLM calls go through this class."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        try:
            self._request_timeout_sec = max(5, int(os.environ.get("OLLAMA_REQUEST_TIMEOUT_SEC", "45")))
        except (TypeError, ValueError):
            self._request_timeout_sec = 45
        try:
            # Увеличиваем контекст, чтобы избежать обрезания
            self._num_ctx = max(1024, int(os.environ.get("OLLAMA_NUM_CTX", "16384")))
        except (TypeError, ValueError):
            self._num_ctx = 16384

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Best-effort conversion for JSON serialization of dynamic payloads."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [LLMClient._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [LLMClient._json_safe(v) for v in value]
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                out[str(k)] = LLMClient._json_safe(v)
            return out
        return str(value)

    def _post_chat_payload(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST payload to Ollama endpoint with serialization fallback."""
        url = f"{self._base_url.rstrip('/')}{endpoint}"
        try:
            body = json.dumps(payload)
        except TypeError:
            safe_payload = self._json_safe(payload)
            body = json.dumps(safe_payload)
            log.warning("LLM payload contained non-serializable objects; coerced to JSON-safe values")

        log.debug("Sending to %s: %s", endpoint, body[:500])
        resp = requests.post(
            url,
            headers=self._headers(),
            data=body,
            timeout=(10, self._request_timeout_sec),
        )
        resp.raise_for_status()
        return resp.json()

    def warmup(self, model: str) -> None:
        """
        Загружает модель с нужным контекстом через /api/chat.
        Выполняется до основного цикла, чтобы модель была в памяти с правильным num_ctx.
        """
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "num_ctx": self._num_ctx,
                },
            }
            self._post_chat_payload("/api/chat", payload)
            log.info(f"Model {model} warmed up with context {self._num_ctx}")
        except Exception as e:
            log.warning(f"Warmup failed for {model}: {e}")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Вызов LLM.
        Если есть tools — используем /v1/chat/completions (openai-совместимый),
        иначе — /api/chat (надёжнее принимает num_ctx).
        Перед вызовом с инструментами рекомендуется выполнить warmup для загрузки модели.
        """
        if tools:
            # Для инструментов используем openai-совместимый эндпоинт
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": self._num_ctx,
                },
                "tools": tools,
            }
            if tool_choice and tool_choice != "auto":
                payload["tool_choice"] = tool_choice
            data = self._post_chat_payload("/v1/chat/completions", payload)
            usage = data.get("usage") or {}
            choices = data.get("choices") or [{}]
            msg = (choices[0] if choices else {}).get("message") or {}
            usage.setdefault("cost", 0.0)
            return msg, usage
        else:
            # Без инструментов используем /api/chat — он гарантированно принимает num_ctx
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": self._num_ctx,
                },
            }
            data = self._post_chat_payload("/api/chat", payload)
            message = data.get("message") or {}
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
                "cost": 0.0,
            }
            return message, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "llava:latest",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({"type": "image_url", "image_url": {"url": img["url"]}})
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img['base64']}"}})
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", "qwen2.5:14b")

    def available_models(self) -> List[str]:
        configured: List[str] = []
        for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_CODE", "OUROBOROS_MODEL_LIGHT"):
            val = os.environ.get(key, "").strip()
            if val and val not in configured:
                configured.append(val)

        discovered = fetch_ollama_models(self._base_url)
        for model in discovered:
            if model not in configured:
                configured.append(model)

        if not configured:
            configured = ["qwen2.5:14b"]
        return configured
