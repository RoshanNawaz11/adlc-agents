"""Agent orchestrator coordinating LLM inference and MCP tool calls."""

from __future__ import annotations

import json
import logging

from .llm_client import LLMClient
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to MCP tools for employee and sales order data. "
    "Use tools when the user asks for specific data. Answer directly for greetings or help requests."
)


class Orchestrator:
    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient):
        """Initialize orchestrator with LLM and MCP clients."""
        self.llm_client = llm_client
        self.mcp_client = mcp_client

    def handle_query(self, query: str) -> str:
        """Process a user query via the LLM and, when needed, MCP tools.

        Returns the final answer string.
        Raises ValueError on argument validation failures.
        Raises RuntimeError when a direct answer or synthesis response is empty,
        or when the synthesis step requests tools again.
        Raises the underlying exception on LLM or MCP failures.
        """
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        tools = self.mcp_client.get_tools()
        response = self.llm_client.infer(messages, tools=tools)
        logger.debug("Initial LLM response: %s", response)

        if response["type"] == "text":
            if not response.get("content", "").strip():
                raise RuntimeError("LLM direct answer returned empty text")
            return response["content"]
        if response["type"] != "tool_use":
            raise RuntimeError(f"Unexpected LLM response type: {response['type']}")

        tool_results = []
        for call in response["calls"]:
            tool_schema = next((tool for tool in tools if tool["name"] == call["name"]), None)
            if not tool_schema:
                raise ValueError(f"LLM requested unknown tool: {call['name']}")

            required = tool_schema["input_schema"].get("required", [])
            for field in required:
                if field not in call["input"]:
                    raise ValueError(f"Missing required field '{field}' for tool '{call['name']}'")

            normalized_input = {
                key: value.upper() if key == "id" else value
                for key, value in call["input"].items()
            }
            logger.info(
                "Invoking MCP tool %s; id normalized from %r to %r; arguments %s",
                call["name"],
                call["input"].get("id"),
                normalized_input.get("id"),
                normalized_input,
            )
            result = self.mcp_client.call_tool(call["name"], normalized_input, timeout=10)
            tool_results.append(
                {
                    "tool_use_id": call["id"],
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "assistant", "content": response["calls"]})
        messages.append({"role": "user", "content": tool_results})
        synthesis_response = self.llm_client.infer(messages, tools=None)
        logger.debug("Synthesis LLM response: %s", synthesis_response)
        if synthesis_response["type"] != "text":
            raise RuntimeError("LLM requested tools again on synthesis step (synthesis loop detected)")
        if not synthesis_response.get("content", "").strip():
            raise RuntimeError("LLM synthesis returned empty text")
        return synthesis_response["content"]
