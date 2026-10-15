"""Integration test with mocked LLM and MCP services."""

from app.agent.http_server import create_app
from app.agent.orchestrator import Orchestrator

EMPLOYEE_TOOL = {
    "name": "get_employee_by_id",
    "description": "Fetch employee details by ID",
    "input_schema": {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def infer(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        query = ""
        has_tool_result = False
        for message in messages:
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, str):
                query = content
            if message.get("role") == "user" and isinstance(content, list):
                has_tool_result = True
        if has_tool_result:
            return {"type": "text", "content": "The name is Alice."}
        if "emp-001" in query:
            return {
                "type": "tool_use",
                "calls": [
                    {
                        "id": "call-1",
                        "name": "get_employee_by_id",
                        "input": {"id": "emp-001"},
                    }
                ],
            }
        return {"type": "text", "content": "Hello!"}


class FakeMCP:
    def __init__(self):
        self.calls = []

    def get_tools(self):
        return [EMPLOYEE_TOOL]

    def call_tool(self, name, arguments, timeout=10):
        self.calls.append({"name": name, "arguments": arguments, "timeout": timeout})
        return {"name": "Alice", "id": "EMP-001"}

    def is_alive(self):
        return True


def test_end_to_end_with_mocked_services():
    llm = FakeLLM()
    mcp = FakeMCP()
    app = create_app(Orchestrator(llm, mcp))

    with app.test_client() as client:
        response = client.post(
            "/query",
            json={"query": "what is the name of employee with emp-001"},
        )

    assert response.status_code == 200
    assert response.get_json() == {"answer": "The name is Alice."}
    assert len(llm.calls) == 2
    assert len(mcp.calls) == 1
    assert mcp.calls[0]["name"] == "get_employee_by_id"
    assert mcp.calls[0]["arguments"] == {"id": "EMP-001"}
