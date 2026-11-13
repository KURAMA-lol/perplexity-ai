"""Vercel ASGI entrypoint for the Perplexity MCP server.

Vercel's function filesystem is read-only except for /tmp. Redirect the
library's state/log directory there before importing the package.
"""
import os

os.environ.setdefault("XDG_STATE_HOME", "/tmp/.local/state")

from mcp.server.transport_security import TransportSecuritySettings

from perplexity.mcp import _register_tools, mcp as perplexity_mcp

if perplexity_mcp is None:
    raise RuntimeError("The MCP dependency is not installed")

# The library's CLI registers tools in main(); an ASGI import must do this too.
# Anonymous mode exposes the auto-mode ask tool. Set PERPLEXITY_COOKIES in
# Vercel environment variables to enable authenticated tools in a later update.
_register_tools(perplexity_mcp, authenticated=False)

app = perplexity_mcp.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
