# adlc-agents

HTTP agent that answers a natural-language question directly, or by calling PER-6 MCP tools for employee and sales-order data.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
export GRAPHQL_ENDPOINT="http://localhost:4000/graphql"
export MCP_SERVER_WORKING_DIR="/path/to/per6/repo"

python -m app.agent.http_server
```

The process listens on `0.0.0.0:5000`.

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}'
```

Success responses are `{"answer": "<text>"}`. `GET /health` returns `{"status": "ok", "mcp_alive": true}` while the MCP subprocess is running.

Employee and sales-order `id` arguments are uppercased before the tool call (`emp-001` becomes `EMP-001`).

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for the LLM client |
| `ANTHROPIC_BASE_URL` | No | `https://api.anthropic.com` | Anthropic API base URL (override for a gateway); empty is treated as unset |
| `ANTHROPIC_MODEL` | No | `claude-3-5-sonnet-20241022` | Model passed to the Messages API; empty is treated as unset |
| `GRAPHQL_ENDPOINT` | Yes | — | GraphQL service URL passed to the MCP subprocess (for example `http://localhost:4000/graphql`) |
| `MCP_SERVER_COMMAND` | No | `python -m mcp.server` | Command used to start the MCP server |
| `MCP_SERVER_WORKING_DIR` | Yes | — | Working directory for the MCP subprocess (PER-6 repo path) |
| `FLASK_HOST` | No | `0.0.0.0` | HTTP server bind host |
| `FLASK_PORT` | No | `5000` | HTTP server bind port |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

`GRAPHQL_ENDPOINT` is forwarded to the MCP subprocess. The agent does not depend on the `mcp` Python package; it speaks JSON-RPC over the server's stdin/stdout.

## Tests

Unit and integration tests mock the LLM and MCP server:

```bash
pip install -r requirements.txt "pytest>=8.0.0"
pytest
```

That run is 21 unit tests (6 HTTP server, 3 LLM client, 4 MCP client, 8 orchestrator) plus 1 integration test.

Manual checks against live PER-5 and PER-6 services are documented in `tests/manual_verification.md`.
