"""Unit tests for the Anthropic LLM client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.agent.llm_client import LLMClient


def test_infer_direct_answer(monkeypatch):
    messages_api = MagicMock()
    messages_api.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello!")]
    )
    _patch_anthropic(monkeypatch, messages_api)

    client = LLMClient(api_key="test-key", base_url="https://example.invalid")
    result = client.infer(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
    )

    assert result == {"type": "text", "content": "Hello!"}
    kwargs = messages_api.create.call_args.kwargs
    assert kwargs["system"] == "You are helpful."
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in kwargs


def test_infer_tool_call(monkeypatch):
    messages_api = MagicMock()
    messages_api.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_123",
                name="get_employee_by_id",
                input={"id": "emp-001"},
            )
        ]
    )
    _patch_anthropic(monkeypatch, messages_api)
    tools = [
        {
            "name": "get_employee_by_id",
            "description": "Fetch employee details by ID",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        }
    ]

    client = LLMClient(api_key="test-key")
    result = client.infer([{"role": "user", "content": "who is emp-001"}], tools=tools)

    assert result == {
        "type": "tool_use",
        "calls": [
            {"id": "toolu_123", "name": "get_employee_by_id", "input": {"id": "emp-001"}}
        ],
    }
    assert messages_api.create.call_args.kwargs["tools"] == tools


def test_infer_api_error(monkeypatch):
    messages_api = MagicMock()
    messages_api.create.side_effect = RuntimeError("rate limit")
    _patch_anthropic(monkeypatch, messages_api)

    client = LLMClient(api_key="test-key")

    with pytest.raises(RuntimeError, match="rate limit"):
        client.infer([{"role": "user", "content": "hello"}])


def _patch_anthropic(monkeypatch, messages_api):
    class FakeAnthropic:
        def __init__(self, api_key, base_url="https://api.anthropic.com"):
            self.api_key = api_key
            self.base_url = base_url
            self.messages = messages_api

    monkeypatch.setattr("app.agent.llm_client.anthropic.Anthropic", FakeAnthropic)
