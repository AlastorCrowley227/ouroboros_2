"""Multi-model review tool using local Ollama models."""

import os
import json
import asyncio
import logging
import httpx

from ouroboros.utils import utc_now_iso
from ouroboros.tools.registry import ToolEntry, ToolContext


log = logging.getLogger(__name__)
MAX_MODELS = 10
CONCURRENCY_LIMIT = 5


def _ollama_chat_url() -> str:
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    return f"{base}/v1/chat/completions"


def get_tools():
    return [
        ToolEntry(
            name="multi_model_review",
            schema={
                "name": "multi_model_review",
                "description": (
                    "Send code or text to multiple local Ollama models for review/consensus. "
                    "Each model reviews independently. Returns structured verdicts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The code or text to review"},
                        "prompt": {"type": "string", "description": "Review instructions."},
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ollama model identifiers (e.g. ['qwen2.5:14b', 'llama3.1:8b'])",
                        },
                    },
                    "required": ["content", "prompt", "models"],
                },
            },
            handler=_handle_multi_model_review,
        )
    ]


def _handle_multi_model_review(ctx: ToolContext, content: str = "", prompt: str = "", models: list = None) -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _multi_model_review_async(content, prompt, models, ctx)).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


async def _query_model(client, model, messages, semaphore):
    async with semaphore:
        try:
            resp = await client.post(
                _ollama_chat_url(),
                headers={"Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.2},
                timeout=120.0,
            )
            status_code = resp.status_code
            response_text = resp.text
            response_headers = dict(resp.headers)
            if status_code != 200:
                error_text = response_text[:200] + (" [truncated]" if len(response_text) > 200 else "")
                return model, f"HTTP {status_code}: {error_text}", None
            return model, resp.json(), response_headers
        except asyncio.TimeoutError:
            return model, "Error: Timeout after 120s", None
        except Exception as e:
            error_msg = str(e)[:200] + (" [truncated]" if len(str(e)) > 200 else "")
            return model, f"Error: {error_msg}", None


async def _multi_model_review_async(content: str, prompt: str, models: list, ctx: ToolContext):
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required (e.g. ['qwen2.5:14b', 'llama3.1:8b'])"}
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return {"error": "models must be a list of strings"}
    if len(models) > MAX_MODELS:
        return {"error": f"Too many models requested ({len(models)}). Maximum is {MAX_MODELS}."}

    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_query_model(client, m, messages, semaphore) for m in models])

    review_results = []
    for model, result, headers_dict in results:
        review_result = _parse_model_response(model, result, headers_dict)
        _emit_usage_event(review_result, ctx)
        review_results.append(review_result)

    return {"model_count": len(models), "results": review_results}


def _parse_model_response(model: str, result, headers_dict) -> dict:
    if isinstance(result, str):
        return {"model": model, "verdict": "ERROR", "text": result, "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0}

    try:
        choices = result.get("choices", [])
        if not choices:
            text = f"(no choices in response: {json.dumps(result)[:200]})"
            verdict = "ERROR"
        else:
            text = choices[0]["message"]["content"]
            verdict = "UNKNOWN"
            for line in text.split("\n")[:3]:
                line_upper = line.upper()
                if "PASS" in line_upper:
                    verdict = "PASS"
                    break
                if "FAIL" in line_upper:
                    verdict = "FAIL"
                    break
    except (KeyError, IndexError, TypeError):
        error_text = json.dumps(result)[:200] + (" [truncated]" if len(json.dumps(result)) > 200 else "")
        text = f"(unexpected response format: {error_text})"
        verdict = "ERROR"

    usage = result.get("usage", {})
    return {
        "model": model,
        "verdict": verdict,
        "text": text,
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "cost_estimate": 0.0,
    }


def _emit_usage_event(review_result: dict, ctx: ToolContext) -> None:
    if ctx is None:
        return
    usage_event = {
        "type": "llm_usage",
        "ts": utc_now_iso(),
        "task_id": ctx.task_id if ctx.task_id else "",
        "usage": {
            "prompt_tokens": review_result["tokens_in"],
            "completion_tokens": review_result["tokens_out"],
            "cost": review_result["cost_estimate"],
        },
        "category": "review",
    }
    if ctx.event_queue is not None:
        try:
            ctx.event_queue.put_nowait(usage_event)
        except Exception:
            if hasattr(ctx, "pending_events"):
                ctx.pending_events.append(usage_event)
    elif hasattr(ctx, "pending_events"):
        ctx.pending_events.append(usage_event)
