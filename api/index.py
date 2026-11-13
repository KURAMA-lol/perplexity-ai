"""Vercel ASGI entrypoint for the Perplexity MCP server.

Vercel's Python function is mounted at /api, so the MCP Streamable HTTP
transport must use that same ASGI path. State files are redirected to /tmp.
Authenticated tools are enabled when PERPLEXITY_COOKIES contains valid JSON.
"""
import os

os.environ.setdefault("XDG_STATE_HOME", "/tmp/.local/state")

from mcp.server.transport_security import TransportSecuritySettings

from perplexity.mcp import _get_client, _register_tools, mcp as perplexity_mcp

if perplexity_mcp is None:
    raise RuntimeError("The MCP dependency is not installed")

# Initialize once per server instance so tool registration matches the
# configured Perplexity session. Without cookies, only perplexity_ask is exposed.
perplexity_client = _get_client()
_register_tools(perplexity_mcp, authenticated=perplexity_client.own)

app = perplexity_mcp.streamable_http_app(
    streamable_http_path="/api",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
