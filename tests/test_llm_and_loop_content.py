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


def test_llm_chat_sets_num_ctx_option(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    captured = {}

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def fake_post(url, headers, data, timeout):
        captured["url"] = url
        captured["payload"] = json.loads(data)
        return DummyResp()

    monkeypatch.setattr("requests.post", fake_post)

    llm = LLMClient(base_url="http://127.0.0.1:11434")
    msg, usage = llm.chat(messages=[{"role": "user", "content": "hi"}], model="llama3.2:latest")

    assert msg["content"] == "ok"
    assert usage["total_tokens"] == 2
    assert captured["payload"]["options"]["num_ctx"] == 8192
    assert captured["url"].endswith("/v1/chat/completions")
