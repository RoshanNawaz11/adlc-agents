# Manual verification

Verify the agent against a real PER-5 GraphQL service and PER-6 MCP server. This satisfies the AC6 manual verification requirement. Automated tests mock those services; this script does not.

## Prerequisites

- PER-5 GraphQL service running at `http://localhost:4000/graphql`
- PER-6 MCP server repository cloned at `/path/to/per6/repo`
- PER-6 virtualenv activated (it provides the `mcp` package)
- Anthropic API key

## Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GRAPHQL_ENDPOINT="http://localhost:4000/graphql"
export MCP_SERVER_WORKING_DIR="/path/to/per6/repo"
export LOG_LEVEL="DEBUG"  # Optional: verbose logging

cd /home/goos/projects/adlc-agents
python -m app.agent.http_server
```

`MCP_SERVER_COMMAND` defaults to `python -m mcp.server`. Override it when the PER-6 entrypoint differs.

## Test cases

### Test 1: AC1/AC2 — Direct answer (no tool)

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}'
```

Expected output:

```json
{"answer": "Hello! How can I help you today?"}
```

Exact wording may vary. The answer should be a greeting, and the logs should show no MCP tool call.

### Test 2: AC3/AC5 — Tool call with ID normalization

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the name of employee with emp-001"}'
```

Expected output:

```json
{"answer": "The employee's name is Alice."}
```

The answer should contain the employee name from GraphQL. Logs should show ID normalization `emp-001` → `EMP-001` before `get_employee_by_id`.

### Test 3: AC4 — List employees tool

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list all employees"}'
```

Expected output:

```json
{"answer": "The employees are: Alice, Bob, Charlie."}
```

The answer should list employee names returned by the GraphQL service.

### Test 4: AC4 — Get sales order tool

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "get sales order ord-001"}'
```

Expected output:

```json
{"answer": "Order ORD-001: Total $1500, Customer: Acme Corp, Status: Shipped"}
```

The answer should contain the order details from GraphQL. Logs should show `ord-001` normalized to `ORD-001`.

### Test 5: Health check

```bash
curl http://localhost:5000/health
```

Expected output:

```json
{"status": "ok", "mcp_alive": true}
```

## Troubleshooting

**MCP subprocess not starting**

- Verify `MCP_SERVER_WORKING_DIR` points to the PER-6 repo root
- Verify the PER-6 virtualenv is activated and `python -m mcp.server` runs manually
- Check agent logs for subprocess stderr output

**GraphQL connection refused**

- Verify PER-5 is running: `curl http://localhost:4000/graphql`
- Check that `GRAPHQL_ENDPOINT` matches the PER-5 service URL

**LLM API errors**

- Verify `ANTHROPIC_API_KEY` is valid
- Check network access to `https://api.anthropic.com` (or the custom `ANTHROPIC_BASE_URL`)
- Review agent logs for API error details

**Incorrect answers**

- Set `LOG_LEVEL=DEBUG` to see LLM prompts, responses, and tool calls
- Verify PER-5 GraphQL data matches the expected values
- Check that MCP tool schemas match the expected format

## Expected timing

- Direct answer queries (Test 1): under 2 seconds
- Tool call queries (Tests 2–4): under 5 seconds per query (LLM inference plus one MCP round-trip)
