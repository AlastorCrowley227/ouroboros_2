import json

from ouroboros.llm import LLMClient
from ouroboros.loop import _handle_text_response, _message_content_to_text


def test_message_content_to_text_handles_block_list():
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        "world",
    ]
    assert _message_content_to_text(content) == "Hello\nworld"


def test_handle_text_response_normalizes_non_string_content():
    text, usage, trace = _handle_text_response(
        [{"type": "text", "text": "Tool JSON as text"}],
        {"assistant_notes": []},
        {},
    )
    assert text == "Tool JSON as text"
    assert trace["assistant_notes"] == ["Tool JSON as text"]
    assert usage == {}


def test_llm_chat_warms_up_then_calls_v1(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    calls = []

    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            if "tools" in self._payload:
                return {
                    "choices": [{"message": {"content": "ok", "tool_calls": []}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            if self._payload.get("messages", [{}])[0].get("content") == "warmup":
                return {"message": {"content": "ready"}, "prompt_eval_count": 2, "eval_count": 1}
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, data, timeout):
        payload = json.loads(data)
        calls.append((url, payload))
        return DummyResp(payload)

    monkeypatch.setattr("requests.post", fake_post)

    llm = LLMClient(base_url="http://127.0.0.1:11434")
    msg, usage = llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="llama3.2:latest",
        tools=[{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
    )

    assert msg["content"] == "ok"
    assert usage["total_tokens"] == 2
    assert calls[0][0].endswith("/api/chat")
    assert calls[0][1]["options"]["num_ctx"] == 8192
    assert calls[1][0].endswith("/v1/chat/completions")


def test_llm_chat_hybrid_uses_native_without_tools(monkeypatch):
    monkeypatch.setenv("OLLAMA_ENDPOINT_STRATEGY", "hybrid")

    calls = []

    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            if self._payload.get("messages", [{}])[0].get("content") == "warmup":
                return {"message": {"content": "ready"}, "prompt_eval_count": 2, "eval_count": 1}
            return {"message": {"content": "ok"}, "prompt_eval_count": 5, "eval_count": 3}

    def fake_post(url, headers, data, timeout):
        payload = json.loads(data)
        calls.append(url)
        return DummyResp(payload)

    monkeypatch.setattr("requests.post", fake_post)

    llm = LLMClient(base_url="http://127.0.0.1:11434")
    msg, usage = llm.chat(messages=[{"role": "user", "content": "hi"}], model="llama3.2:latest")

    assert msg["content"] == "ok"
    assert usage["prompt_tokens"] == 5
    assert calls[0].endswith("/api/chat")
    assert calls[1].endswith("/api/chat")
