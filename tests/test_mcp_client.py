"""Unit tests for the MCP stdio client."""

import json
import os
import select
import sys
import textwrap

import pytest

from app.agent.mcp_client import MCPClient

FOUR_TOOLS = [
    {
        "name": "list_sales_order",
        "description": "List sales orders",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_employees",
        "description": "List employees",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_sales_order_by_id",
        "description": "Get a sales order by ID",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "get_employee_by_id",
        "description": "Fetch employee details by ID",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
]


@pytest.fixture
def mcp_factory(tmp_path):
    clients: list[MCPClient] = []

    def make(script: str) -> MCPClient:
        path = tmp_path / f"server_{len(clients)}.py"
        path.write_text(script)
        env = dict(os.environ)
        env["GRAPHQL_ENDPOINT"] = "http://localhost:4000/graphql"
        client = MCPClient(f"{sys.executable} -u {path}", str(tmp_path), env)
        clients.append(client)
        return client

    yield make

    for client in clients:
        client.shutdown()


def test_start_successful_handshake(mcp_factory):
    client = mcp_factory(_success_script())

    client.start()
    tools = client.get_tools()

    assert {tool["name"] for tool in tools} == {tool["name"] for tool in FOUR_TOOLS}
    employee = next(tool for tool in tools if tool["name"] == "get_employee_by_id")
    assert employee["input_schema"]["required"] == ["id"]
    assert "inputSchema" not in employee
    assert all("input_schema" in tool and tool["description"] for tool in tools)


def test_start_handshake_failure(mcp_factory):
    client = mcp_factory(_failure_script())

    with pytest.raises(RuntimeError, match="MCP handshake failed:") as exc_info:
        client.start()

    assert "boom" in str(exc_info.value)


def test_shutdown_graceful(mcp_factory):
    client = mcp_factory(_success_script())
    client.start()
    assert client.is_alive()

    killed = {"called": False}

    def fail_kill():
        killed["called"] = True
        raise AssertionError("SIGKILL should not be required")

    client.process.kill = fail_kill
    client.shutdown()

    assert killed["called"] is False
    assert client.process.returncode == 0
    assert client.process.stdin.closed
    assert client.is_alive() is False


def test_call_tool_timeout(mcp_factory, monkeypatch):
    client = mcp_factory(_success_script())
    client.start()

    timeouts = []

    def not_ready(read_list, write_list, error_list, timeout):
        timeouts.append(timeout)
        return [], [], []

    monkeypatch.setattr(select, "select", not_ready)

    with pytest.raises(TimeoutError) as exc_info:
        client.call_tool("get_employee_by_id", {"id": "EMP-001"}, timeout=10)

    assert str(exc_info.value) == "MCP tool call 'get_employee_by_id' timed out after 10s"
    assert timeouts == [10]


def _success_script() -> str:
    tools = json.dumps(FOUR_TOOLS)
    return textwrap.dedent(
        f"""
        import json
        import sys

        TOOLS = json.loads({tools!r})

        def read_request():
            line = sys.stdin.readline()
            if not line:
                return None
            return json.loads(line)

        def write(message):
            sys.stdout.write(json.dumps(message) + "\\n")
            sys.stdout.flush()

        request = read_request()
        write({{
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {{"protocolVersion": "2024-11-05", "capabilities": {{}}}},
        }})
        request = read_request()
        while request is not None and request.get("method") == "notifications/initialized":
            request = read_request()
        write({{
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {{"tools": TOOLS}},
        }})
        sys.stdin.read()
        """
    )


def _failure_script() -> str:
    return textwrap.dedent(
        """
        import json
        import sys

        request = json.loads(sys.stdin.readline())
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": -32000, "message": "boom"},
        }) + "\\n")
        sys.stdout.flush()
        """
    )
