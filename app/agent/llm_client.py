"""Anthropic LLM client for inference and tool calling."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = DEFAULT_MODEL,
    ):
        """Initialize Anthropic client with API credentials."""
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def infer(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Call the Anthropic Messages API.

        Args:
            messages: Conversation history (system, user, assistant, tool_result).
            tools: Tool schemas in Anthropic format (optional).

        Returns:
            Parsed response:
                {"type": "text", "content": "<answer>"}
                OR
                {"type": "tool_use", "calls": [{"id": "<id>", "name": "<name>", "input": {...}}, ...]}

        Raises:
            Exception on API errors (network, auth, rate limit).
        """
        system, chat_messages = _convert_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system:
            request["system"] = system
        if tools is not None:
            request["tools"] = tools
        logger.debug("Anthropic request: %s", request)
        response = self._client.messages.create(**request)
        parsed = _parse_response(response)
        logger.debug("Anthropic parsed response: %s", parsed)
        return parsed


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        if role == "assistant" and isinstance(content, list):
            converted.append({"role": "assistant", "content": [_assistant_block(item) for item in content]})
            continue
        if role == "user" and isinstance(content, list):
            converted.append({"role": "user", "content": [_user_block(item) for item in content]})
            continue
        converted.append({"role": role, "content": content})
    system = "\n".join(part for part in system_parts if part) or None
    return system, converted


def _assistant_block(item: Any) -> dict:
    if isinstance(item, dict) and "name" in item and "input" in item:
        return {
            "type": "tool_use",
            "id": item.get("id", ""),
            "name": item["name"],
            "input": item["input"],
        }
    return item


def _user_block(item: Any) -> dict:
    if isinstance(item, dict) and "tool_use_id" in item:
        return {
            "type": "tool_result",
            "tool_use_id": item["tool_use_id"],
            "content": item.get("content", ""),
        }
    return item


def _parse_response(response: Any) -> dict:
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
    if not content:
        raise RuntimeError("LLM response did not include content")

    calls = []
    texts: list[str] = []
    for block in content:
        block_type = _block_value(block, "type")
        if block_type == "tool_use":
            calls.append(
                {
                    "id": _block_value(block, "id"),
                    "name": _block_value(block, "name"),
                    "input": _block_value(block, "input") or {},
                }
            )
        elif block_type == "text":
            texts.append(_block_value(block, "text") or "")

    if calls:
        return {"type": "tool_use", "calls": calls}
    if texts:
        return {"type": "text", "content": "".join(texts)}
    raise RuntimeError("LLM response did not include text or tool calls")


def _block_value(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)
