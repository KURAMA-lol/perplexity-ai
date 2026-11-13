"""Vercel ASGI entrypoint exposing MCP and an OpenAI-compatible REST API."""
import asyncio
import hmac
import json
import logging
import os
import threading
import time
from uuid import uuid4

os.environ.setdefault("XDG_STATE_HOME", "/tmp/.local/state")

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from perplexity.config import MODEL_MAPPINGS
from perplexity.exceptions import AuthenticationError, RateLimitError
from perplexity.mcp import _extract_answer, _get_client, _register_tools, mcp as perplexity_mcp

logger = logging.getLogger("perplexity.api")
if perplexity_mcp is None:
    raise RuntimeError("The MCP dependency is not installed")

perplexity_client = _get_client()
_register_tools(perplexity_mcp, authenticated=perplexity_client.own)
mcp_app = perplexity_mcp.streamable_http_app(
    streamable_http_path="/api",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
_provider_lock = threading.Lock()


def _auth_error(request: Request):
    expected = os.environ.get("PERPLEXITY_API_KEY")
    if not expected:
        return JSONResponse({"error": {"message": "Server is not configured: PERPLEXITY_API_KEY is missing.",
                                       "type": "server_error", "code": "api_key_not_configured"}}, status_code=503)
    scheme, separator, supplied = request.headers.get("authorization", "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip(), expected):
        return JSONResponse({"error": {"message": "Invalid or missing API key.",
                                       "type": "authentication_error", "code": "invalid_api_key"}},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    return None


def _select_model(model):
    aliases = {
        "perplexity": ("auto", None), "auto": ("auto", None), "pplx-auto": ("auto", None),
        "pro": ("pro", None), "pplx-pro": ("pro", None), "sonar": ("pro", "sonar"),
        "reasoning": ("reasoning", None), "pplx-reasoning": ("reasoning", None),
        "deep-research": ("deep research", None), "deep_research": ("deep research", None),
        "deep research": ("deep research", None),
    }
    model = model or "perplexity"
    if model in aliases:
        return aliases[model]
    for mode, mapping in MODEL_MAPPINGS.items():
        if model in mapping and model is not None:
            return mode, model
    raise ValueError("Unsupported model ID: " + str(model))


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
                raise ValueError("Only text message content is supported.")
            parts.append(str(part.get("text", "")))
        return "".join(parts)
    raise ValueError("Message content must be a string or an array of text parts.")


def _messages_to_query(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array.")
    if len(messages) > 100:
        raise ValueError("A maximum of 100 messages is supported.")
    allowed = {"system", "developer", "user", "assistant"}
    lines = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be an object.")
        role = item.get("role")
        if role not in allowed:
            raise ValueError("Supported roles: system, developer, user, assistant.")
        text = _content_text(item.get("content", ""))
        if text:
            lines.append(role.capitalize() + ": " + text)
    if not lines:
        raise ValueError("messages contains no text.")
    return "\n\n".join(lines)


def _error(message, error_type="invalid_request_error", code=None, status=400):
    return JSONResponse({"error": {"message": message, "type": error_type, "code": code}}, status_code=status)


async def list_models(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    ids = {"perplexity", "auto", "pplx-auto", "pro", "pplx-pro", "sonar",
           "reasoning", "pplx-reasoning", "deep-research", "deep_research"}
    for mapping in MODEL_MAPPINGS.values():
        ids.update(name for name in mapping if name is not None)
    return JSONResponse({"object": "list", "data": [
        {"id": name, "object": "model", "created": 0, "owned_by": "perplexity"}
        for name in sorted(ids)
    ]})


async def chat_completions(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    try:
        raw = await request.body()
        if len(raw) > 1_000_000:
            return _error("Request body exceeds 1 MB.", status=413)
        payload = json.loads(raw or b"{}")
        if not isinstance(payload, dict):
            return _error("Request body must be a JSON object.")
        if payload.get("tools") or payload.get("tool_choice"):
            return _error("Tool calling is not supported by the underlying Perplexity web client.",
                          code="unsupported_parameter")
        query = _messages_to_query(payload.get("messages"))
        mode, provider_model = _select_model(payload.get("model", "perplexity"))
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            return _error("stream must be a boolean.")
        completion_id, created = "chatcmpl-" + uuid4().hex, int(time.time())
        public_model = payload.get("model") or "perplexity"

        if stream:
            def generate():
                previous = ""
                try:
                    with _provider_lock:
                        chunks = perplexity_client.search(query, mode=mode, model=provider_model, stream=True)
                        for chunk in chunks:
                            current = _extract_answer(chunk)
                            if not current:
                                continue
                            delta = current[len(previous):] if current.startswith(previous) else current
                            previous = current
                            if delta:
                                event = {"id": completion_id, "object": "chat.completion.chunk",
                                         "created": created, "model": public_model,
                                         "choices": [{"index": 0, "delta": {"content": delta},
                                                       "finish_reason": None}]}
                                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    event = {"id": completion_id, "object": "chat.completion.chunk",
                             "created": created, "model": public_model,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield "data: " + json.dumps(event) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    logger.warning("Provider stream failed (%s)", exc.__class__.__name__)
                    yield "data: " + json.dumps({"error": {"message": "Perplexity request failed.",
                                                           "type": "server_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            with _provider_lock:
                result = await asyncio.to_thread(perplexity_client.search, query,
                                                 mode=mode, model=provider_model, stream=False)
            answer = _extract_answer(result)
        except AuthenticationError:
            return _error("Perplexity session authentication failed.", "authentication_error",
                          "provider_authentication_failed", 502)
        except RateLimitError:
            return _error("Perplexity rate limit reached; try again later.",
                          "rate_limit_error", "provider_rate_limit", 429)
        except Exception as exc:
            logger.warning("Provider request failed (%s)", exc.__class__.__name__)
            return _error("Perplexity request failed.", "server_error", "provider_error", 502)
        if not answer:
            return _error("Perplexity returned an empty answer.", "server_error",
                          "empty_provider_response", 502)
        return JSONResponse({
            "id": completion_id, "object": "chat.completion", "created": created,
            "model": public_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    except json.JSONDecodeError:
        return _error("Request body must contain valid JSON.")
    except ValueError as exc:
        return _error(str(exc))


async def health(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    return JSONResponse({"status": "ok", "provider": "perplexity"})


api_app = Starlette(routes=[
    Route("/api/v1/models", list_models, methods=["GET", "HEAD"]),
    Route("/api/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/api/v1/health", health, methods=["GET", "HEAD"]),
])


async def app(scope, receive, send):
    if scope["type"] == "http":
        path = scope.get("path", "")
        if path.startswith("/api/v1/"):
            await api_app(scope, receive, send)
            return
        if path == "/api" or path.startswith("/api/"):
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            expected = os.environ.get("PERPLEXITY_API_KEY")
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            scheme, separator, supplied = authorization.partition(" ")
            valid = bool(expected and separator and scheme.lower() == "bearer"
                         and hmac.compare_digest(supplied.strip(), expected))
            if not expected:
                response = JSONResponse({"error": "PERPLEXITY_API_KEY is missing."}, status_code=503)
                await response(scope, receive, send)
                return
            if not valid:
                response = JSONResponse({"error": "Invalid or missing API key."}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
    await mcp_app(scope, receive, send)
