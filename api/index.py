"""Vercel ASGI entrypoint for the Perplexity MCP server.

Vercel's Python function is mounted at /api, so the MCP Streamable HTTP
transport must use that same ASGI path. State files are redirected to /tmp.
"""
import os

os.environ.setdefault("XDG_STATE_HOME", "/tmp/.local/state")

from mcp.server.transport_security import TransportSecuritySettings

from perplexity.mcp import _register_tools, mcp as perplexity_mcp

if perplexity_mcp is None:
    raise RuntimeError("The MCP dependency is not installed")

_register_tools(perplexity_mcp, authenticated=False)

app = perplexity_mcp.streamable_http_app(
    streamable_http_path="/api",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
