"""Ollama-backed LLM client used by Ouroboros.

Design goals:
- Keep one place responsible for all HTTP contracts with Ollama.
- Guarantee a first warmup call to /api/chat with explicit num_ctx.
- Prefer a single endpoint strategy to avoid accidental model reloads with a
  smaller context window.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "qwen2.5:3b"


class LLMError(RuntimeError):
    """Base error for all LLM client failures."""


class LLMTransportError(LLMError):
    """HTTP-level failure while calling Ollama."""


class LLMProtocolError(LLMError):
    """Unexpected JSON schema from Ollama response."""


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


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
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        log.debug("Failed to fetch Ollama model list", exc_info=True)
        return []


class LLMClient:
    """HTTP wrapper around Ollama with context-safe warmup semantics."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self._request_timeout_sec = self._read_int_env("OLLAMA_REQUEST_TIMEOUT_SEC", 45, min_value=5)
        self._num_ctx = self._read_int_env("OLLAMA_NUM_CTX", 32768, min_value=1024)

        # Endpoint strategy:
        # - single_v1 (default): use /v1/chat/completions for all requests.
        # - hybrid: tools via /v1/chat/completions, plain chat via /api/chat.
        strategy = str(os.environ.get("OLLAMA_ENDPOINT_STRATEGY", "single_v1")).strip().lower()
        self._endpoint_strategy = strategy if strategy in {"single_v1", "hybrid"} else "single_v1"

        # Warmup is tracked per model; first call for each model will load n_ctx.
        self._warmed_models: set[str] = set()

    @staticmethod
    def _read_int_env(name: str, default: int, min_value: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(min_value, value)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [LLMClient._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [LLMClient._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): LLMClient._json_safe(v) for k, v in value.items()}
        return str(value)

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url.rstrip('/')}{endpoint}"
        safe_payload = self._json_safe(payload)
        body = json.dumps(safe_payload)

        try:
            response = requests.post(
                url,
                headers=self._headers(),
                data=body,
                timeout=(10, self._request_timeout_sec),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise LLMTransportError(f"POST {endpoint} failed: {exc}") from exc

    def _warmup_payload(self, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "warmup"}],
            "options": {
                "num_predict": 1,
                "num_ctx": self._num_ctx,
            },
        }

    def ensure_model_ready(self, model: str) -> None:
        """Warm model once with explicit num_ctx using /api/chat."""
        if model in self._warmed_models:
            return

        data = self._post_json("/api/chat", self._warmup_payload(model))
        prompt_eval_count = int(data.get("prompt_eval_count") or 0)
        log.info(
            "Warmup finished for model=%s num_ctx=%s prompt_eval_count=%s",
            model,
            self._num_ctx,
            prompt_eval_count,
        )
        self._warmed_models.add(model)

    # Backward-compatible name used by older call sites.
    def warmup(self, model: str) -> None:
        try:
            self.ensure_model_ready(model)
        except Exception:
            log.warning("Warmup failed for %s", model, exc_info=True)

    def _chat_v1(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
            if tool_choice and tool_choice != "auto":
                payload["tool_choice"] = tool_choice

        data = self._post_json("/v1/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMProtocolError("Missing choices[0] in /v1/chat/completions response")

        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        usage.setdefault("prompt_tokens", int(data.get("prompt_eval_count") or 0))
        usage.setdefault("completion_tokens", int(data.get("eval_count") or 0))
        usage.setdefault("total_tokens", int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
        usage.setdefault("cost", 0.0)
        usage["prompt_eval_count"] = int(data.get("prompt_eval_count") or usage.get("prompt_tokens") or 0)
        return message, usage

    def _chat_native(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        data = self._post_json("/api/chat", payload)
        message = data.get("message") or {}
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
            "total_tokens": int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0),
            "cost": 0.0,
            "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
        }
        return message, usage

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        del reasoning_effort  # kept for API compatibility

        # Guarantee warmup even if first call is with tools.
        self.ensure_model_ready(model)

        if self._endpoint_strategy == "hybrid" and not tools:
            message, usage = self._chat_native(messages, model, max_tokens)
        else:
            message, usage = self._chat_v1(messages, model, max_tokens, tools, tool_choice)

        log.info(
            "LLM response model=%s strategy=%s prompt_eval_count=%s prompt_tokens=%s completion_tokens=%s",
            model,
            self._endpoint_strategy,
            int(usage.get("prompt_eval_count") or 0),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )
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
        return str(response_msg.get("content") or ""), usage

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", "llama3.2-32k:latest")

    def available_models(self) -> List[str]:
        configured: List[str] = []
        for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_CODE", "OUROBOROS_MODEL_LIGHT"):
            value = os.environ.get(key, "").strip()
            if value and value not in configured:
                configured.append(value)

        discovered = fetch_ollama_models(self._base_url)
        for model in discovered:
            if model not in configured:
                configured.append(model)

        return configured or ["llama3.2-32k:latest"]
