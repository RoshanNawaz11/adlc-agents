"""Unit tests for the agent orchestrator."""

from unittest.mock import MagicMock

import pytest

from app.agent.orchestrator import Orchestrator

EMPLOYEE_TOOL = {
    "name": "get_employee_by_id",
    "description": "Fetch employee details by ID",
    "input_schema": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Employee ID (e.g., EMP-001)"}},
        "required": ["id"],
    },
}

TOOL_CALL = {
    "type": "tool_use",
    "calls": [
        {
            "id": "call-1",
            "name": "get_employee_by_id",
            "input": {"id": "emp-001"},
        }
    ],
}


def test_direct_answer_path_and_empty_validation():
    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {"type": "text", "content": "Hello!"}

    assert orchestrator.handle_query("hello") == "Hello!"
    mcp.call_tool.assert_not_called()
    assert llm.infer.call_count == 1

    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {"type": "text", "content": ""}
    with pytest.raises(RuntimeError, match="LLM direct answer returned empty text"):
        orchestrator.handle_query("hello")
    mcp.call_tool.assert_not_called()

    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {"type": "text", "content": "   "}
    with pytest.raises(RuntimeError, match="LLM direct answer returned empty text"):
        orchestrator.handle_query("hello")
    mcp.call_tool.assert_not_called()


def test_tool_call_success_and_synthesis_empty_validation():
    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [
        TOOL_CALL,
        {"type": "text", "content": "The name is Alice."},
    ]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}

    assert orchestrator.handle_query("what is the name of employee with emp-001") == "The name is Alice."
    assert llm.infer.call_count == 2
    assert mcp.call_tool.call_count == 1
    mcp.call_tool.assert_called_once_with("get_employee_by_id", {"id": "EMP-001"}, timeout=10)
    assert llm.infer.call_args_list[1].kwargs["tools"] is None

    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [
        TOOL_CALL,
        {"type": "text", "content": ""},
    ]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}
    with pytest.raises(RuntimeError, match="LLM synthesis returned empty text"):
        orchestrator.handle_query("what is the name of employee with emp-001")

    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [
        TOOL_CALL,
        {"type": "text", "content": "   "},
    ]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}
    with pytest.raises(RuntimeError, match="LLM synthesis returned empty text"):
        orchestrator.handle_query("what is the name of employee with emp-001")


def test_tool_call_mcp_error():
    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = TOOL_CALL
    mcp.call_tool.side_effect = RuntimeError("tool error")

    with pytest.raises(RuntimeError, match="tool error"):
        orchestrator.handle_query("what is the name of employee with emp-001")


def test_llm_call_count_limit():
    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {"type": "text", "content": "Hello!"}
    orchestrator.handle_query("hello")
    assert llm.infer.call_count == 1
    mcp.call_tool.assert_not_called()

    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [
        TOOL_CALL,
        {"type": "text", "content": "The name is Alice."},
    ]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}
    orchestrator.handle_query("what is the name of employee with emp-001")
    assert llm.infer.call_count == 2
    assert mcp.call_tool.call_count == 1


def test_id_normalization():
    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [
        TOOL_CALL,
        {"type": "text", "content": "The name is Alice."},
    ]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}

    orchestrator.handle_query("what is the name of employee with emp-001")

    assert mcp.call_tool.call_args.args[1] == {"id": "EMP-001"}


def test_argument_validation_missing_field():
    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {
        "type": "tool_use",
        "calls": [{"id": "call-1", "name": "get_employee_by_id", "input": {}}],
    }

    with pytest.raises(ValueError) as exc_info:
        orchestrator.handle_query("look up an employee")

    assert str(exc_info.value) == "Missing required field 'id' for tool 'get_employee_by_id'"
    mcp.call_tool.assert_not_called()


def test_argument_validation_unknown_tool():
    llm, mcp, orchestrator = _clients()
    llm.infer.return_value = {
        "type": "tool_use",
        "calls": [{"id": "call-1", "name": "nonexistent_tool", "input": {"id": "emp-001"}}],
    }

    with pytest.raises(ValueError) as exc_info:
        orchestrator.handle_query("do something")

    assert str(exc_info.value) == "LLM requested unknown tool: nonexistent_tool"
    mcp.call_tool.assert_not_called()


def test_synthesis_loop_detection():
    llm, mcp, orchestrator = _clients()
    llm.infer.side_effect = [TOOL_CALL, TOOL_CALL]
    mcp.call_tool.return_value = {"name": "Alice", "id": "EMP-001"}

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.handle_query("what is the name of employee with emp-001")

    assert str(exc_info.value) == (
        "LLM requested tools again on synthesis step (synthesis loop detected)"
    )
    assert llm.infer.call_count == 2


def _clients():
    llm = MagicMock()
    mcp = MagicMock()
    mcp.get_tools.return_value = [EMPLOYEE_TOOL]
    return llm, mcp, Orchestrator(llm, mcp)
