"""Flask HTTP server with /query and /health endpoints."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from .llm_client import LLMClient
from .mcp_client import MCPClient
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

DEFAULT_MCP_COMMAND = "python -m mcp.server"


def create_app(orchestrator: Orchestrator) -> Flask:
    """Create and configure Flask app with /query and /health endpoints."""
    app = Flask(__name__)

    @app.post("/query")
    def query():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "malformed JSON"}), 400
        if (
            not isinstance(payload, dict)
            or "query" not in payload
            or not isinstance(payload.get("query"), str)
        ):
            return jsonify({"error": "missing 'query' field"}), 400
        if not payload["query"].strip():
            return jsonify({"error": "query cannot be empty"}), 400
        try:
            answer = orchestrator.handle_query(payload["query"])
        except ValueError as exc:
            logger.exception("Rejected query")
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("Query failed")
            return jsonify({"error": "internal error"}), 500
        return jsonify({"answer": answer}), 200

    @app.get("/health")
    def health():
        if orchestrator.mcp_client.is_alive():
            return jsonify({"status": "ok", "mcp_alive": True}), 200
        return jsonify({"status": "degraded", "mcp_alive": False}), 503

    return app


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the HTTP server (blocking)."""
    _configure_logging()
    if os.environ.get("FLASK_HOST"):
        host = os.environ["FLASK_HOST"]
    if os.environ.get("FLASK_PORT"):
        port = int(os.environ["FLASK_PORT"])

    api_key = _required_env("ANTHROPIC_API_KEY")
    base_url = (
        os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip()
        or "https://api.anthropic.com"
    )
    model = (
        os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
        or "claude-3-5-sonnet-20241022"
    )

    graphql_endpoint = _required_env("GRAPHQL_ENDPOINT")
    working_dir = _required_env("MCP_SERVER_WORKING_DIR")
    server_command = os.environ.get("MCP_SERVER_COMMAND", DEFAULT_MCP_COMMAND).strip() or DEFAULT_MCP_COMMAND

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model)
    child_env = dict(os.environ)
    child_env["GRAPHQL_ENDPOINT"] = graphql_endpoint
    mcp_client = MCPClient(server_command, working_dir, child_env)

    try:
        mcp_client.start()
    except Exception:
        logger.exception("MCP subprocess failed to start")
        raise

    app = create_app(Orchestrator(llm_client, mcp_client))
    logger.info("Agent listening on %s:%s", host, port)
    try:
        app.run(host=host, port=port, use_reloader=False)
    finally:
        mcp_client.shutdown()


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    run_server()
