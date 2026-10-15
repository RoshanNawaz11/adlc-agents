"""MCP stdio client: subprocess lifecycle, tool discovery, and tool invocation."""

from __future__ import annotations

import json
import logging
import select
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

REQUIRED_TOOLS = {
    "list_sales_order",
    "list_employees",
    "get_sales_order_by_id",
    "get_employee_by_id",
}


class MCPClient:
    def __init__(self, server_command: str, working_dir: str, env: dict[str, str]):
        """Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server (e.g. "python -m mcp.server").
            working_dir: Working directory for the subprocess.
            env: Environment variables for the subprocess (must include GRAPHQL_ENDPOINT).
        """
        self.server_command = server_command
        self.working_dir = working_dir
        self.env = env
        self.process: subprocess.Popen[str] | None = None
        self._tools: list[dict] | None = None
        self._request_id = 1
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the MCP subprocess, perform the JSON-RPC handshake, and fetch tools.

        Raises:
            Exception on handshake failure (agent fails to start).
        """
        if not self.env.get("GRAPHQL_ENDPOINT"):
            raise RuntimeError("MCP environment must include GRAPHQL_ENDPOINT")
        if self.is_alive():
            raise RuntimeError("MCP subprocess is already running")

        self.process = subprocess.Popen(
            self.server_command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.working_dir,
            env=self.env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="mcp-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            self._initialize()
        except Exception:
            self.shutdown()
            raise

    def get_tools(self) -> list[dict]:
        """Return available tool schemas in Anthropic format.

        Must be called after start().
        """
        if self._tools is None:
            raise RuntimeError("MCP client has not been started")
        return self._tools

    def call_tool(self, name: str, arguments: dict, timeout: int = 10) -> dict:
        """Invoke an MCP tool via JSON-RPC over stdio.

        Args:
            name: Tool name (e.g. "get_employee_by_id").
            arguments: Tool input (e.g. {"id": "EMP-001"}).
            timeout: Request timeout in seconds.

        Returns:
            Tool result dict.

        Raises:
            TimeoutError if the request exceeds timeout.
            RuntimeError on an MCP protocol error, tool execution error, or when
            the subprocess emits more than 100 consecutive empty lines.
        """
        if not self.is_alive():
            raise RuntimeError("MCP subprocess is not running")

        request_id = self._next_id()
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        response = self._read_json(
            timeout,
            TimeoutError(f"MCP tool call '{name}' timed out after {timeout}s"),
        )
        if "error" in response:
            raise RuntimeError(f"MCP tool '{name}' returned error: {response['error']}")
        if "result" not in response:
            raise RuntimeError(f"MCP tool '{name}' returned no result")
        return response["result"]

    def is_alive(self) -> bool:
        """Check if the MCP subprocess is still running."""
        return self.process is not None and self.process.poll() is None

    def shutdown(self) -> None:
        """Gracefully terminate the MCP subprocess (EOF, wait 5s, SIGKILL if needed)."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    logger.debug("Failed to close MCP stdin", exc_info=True)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP subprocess did not exit within 5s; sending SIGKILL")
                process.kill()
                process.wait(timeout=5)
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        self._close_pipes(process)

    def _initialize(self) -> None:
        self._write(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                },
            }
        )
        response = self._read_json(5, RuntimeError("MCP handshake timed out after 5s"))
        if "error" in response:
            raise RuntimeError(f"MCP handshake failed: {response['error']}")

        # MCP requires this notification before any other request.
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._write({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        listed = self._read_json(5, RuntimeError("MCP tools/list timed out after 5s"))
        if "error" in listed:
            raise RuntimeError(f"MCP tools/list failed: {listed['error']}")

        tools = listed.get("result", {}).get("tools", [])
        actual_tools = {tool["name"] for tool in tools}
        missing = REQUIRED_TOOLS - actual_tools
        if missing:
            raise RuntimeError(f"Missing required tools: {missing}")
        self._tools = self._convert_to_anthropic_format(tools)
        logger.info("MCP server ready with tools: %s", sorted(actual_tools))

    def _convert_to_anthropic_format(self, tools: list[dict]) -> list[dict]:
        converted = []
        for tool in tools:
            schema = tool.get("input_schema", tool.get("inputSchema", {"type": "object", "properties": {}}))
            converted.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": schema,
                }
            )
        return converted

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _write(self, payload: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("MCP subprocess is not running")
        logger.debug("MCP request: %s", payload)
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_json(self, timeout: float, timeout_error: Exception) -> dict:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("MCP subprocess is not running")

        deadline = time.monotonic() + timeout
        stdout = self.process.stdout
        first_wait = True
        empty_line_count = 0
        while True:
            if first_wait:
                wait = timeout
                first_wait = False
            else:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    raise timeout_error
            ready, _, _ = select.select([stdout], [], [], wait)
            if not ready:
                raise timeout_error
            line = stdout.readline()
            if line == "":
                raise RuntimeError("MCP subprocess closed stdout unexpectedly")
            if not line.strip():
                empty_line_count += 1
                if empty_line_count > 100:
                    raise RuntimeError(
                        "MCP subprocess emitted >100 consecutive empty lines (possible keepalive loop)"
                    )
                continue
            empty_line_count = 0
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON from MCP subprocess: {line!r}") from exc
            logger.debug("MCP response: %s", message)
            if "result" in message or "error" in message:
                return message

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                logger.info("MCP stderr: %s", line.rstrip())
        except (OSError, ValueError):
            return

    @staticmethod
    def _close_pipes(process: subprocess.Popen[str]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is None or stream.closed:
                continue
            try:
                stream.close()
            except OSError:
                logger.debug("Failed to close MCP pipe", exc_info=True)
