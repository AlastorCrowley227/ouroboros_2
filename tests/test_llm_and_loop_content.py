import json
from ouroboros.llm import LLMClient
from ouroboros.loop import _extract_text_tool_call, _handle_text_response, _message_content_to_text


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


def test_llm_chat_sets_num_ctx_option(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    captured = {}

    class DummyResp:
        def __init__(self, url):
            self._url = url

        def raise_for_status(self):
            return None

        def json(self):
            if self._url.endswith("/api/chat"):
                return {
                    "message": {"content": "ok"},
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, data, timeout):
        captured["url"] = url
        captured["payload"] = json.loads(data)
        return DummyResp(url)

    monkeypatch.setattr("requests.post", fake_post)

    llm = LLMClient(base_url="http://127.0.0.1:11434")
    msg, usage = llm.chat(messages=[{"role": "user", "content": "hi"}], model="llama3.2:latest")

    assert msg["content"] == "ok"
    assert usage["total_tokens"] == 2
    assert captured["payload"]["options"]["num_ctx"] == 8192
    assert captured["url"].endswith("/api/chat")


def test_extract_text_tool_call_from_name_and_arguments():
    schemas = [{"type": "function", "function": {"name": "update_identity"}}]
    calls = _extract_text_tool_call(
        '{"name":"update_identity","arguments":{"content":"x"}}',
        schemas,
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "update_identity"
    assert '"content": "x"' in calls[0]["function"]["arguments"]


def test_extract_text_tool_call_from_function_name_alias():
    schemas = [{"type": "function", "function": {"name": "get_task_result"}}]
    calls = _extract_text_tool_call(
        '{"function_name":"get_task_result","arguments":{"task_id":"1"}}',
        schemas,
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_task_result"
