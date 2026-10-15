"""Unit tests for the agent HTTP server."""

import logging

from app.agent.http_server import create_app


def _app(orchestrator):
    return create_app(orchestrator)


def test_query_endpoint_valid_request():
    orchestrator = _orchestrator("Hello there")
    client = _app(orchestrator).test_client()

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 200
    assert response.get_json() == {"answer": "Hello there"}
    orchestrator.handle_query.assert_called_once_with("hello")


def test_query_endpoint_malformed_json():
    orchestrator = _orchestrator("unused")
    client = _app(orchestrator).test_client()

    response = client.post("/query", data="{invalid}", content_type="application/json")

    assert response.status_code == 400
    body = response.get_json()
    assert isinstance(body["error"], str) and body["error"]
    orchestrator.handle_query.assert_not_called()


def test_query_endpoint_missing_query_field():
    orchestrator = _orchestrator("unused")
    client = _app(orchestrator).test_client()

    response = client.post("/query", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "missing 'query' field"}
    orchestrator.handle_query.assert_not_called()


def test_query_endpoint_internal_error(caplog):
    orchestrator = _orchestrator("unused")
    orchestrator.handle_query.side_effect = RuntimeError("llm exploded")
    client = _app(orchestrator).test_client()

    with caplog.at_level(logging.ERROR):
        response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    assert response.get_json() == {"error": "internal error"}
    assert any(record.exc_info for record in caplog.records)
    assert "llm exploded" in caplog.text


def test_health_endpoint_mcp_alive():
    orchestrator = _orchestrator("unused")
    orchestrator.mcp_client.is_alive.return_value = True
    client = _app(orchestrator).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "mcp_alive": True}


def test_health_endpoint_mcp_crashed():
    orchestrator = _orchestrator("unused")
    orchestrator.mcp_client.is_alive.return_value = False
    client = _app(orchestrator).test_client()

    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "degraded", "mcp_alive": False}


def _orchestrator(answer: str):
    from unittest.mock import MagicMock

    orchestrator = MagicMock()
    orchestrator.handle_query.return_value = answer
    return orchestrator
