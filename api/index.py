"""Vercel ASGI entrypoint for the Perplexity MCP server.

Vercel serves this app as a Python Function. Stateless HTTP avoids relying on
an in-memory MCP session being routed to the same function instance.
"""
from mcp.server.transport_security import TransportSecuritySettings

from perplexity.mcp import mcp as perplexity_mcp

app = perplexity_mcp.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
