"""Small authenticated OpenAI-compatible proxy for the user's Gemini Web API."""
import hmac
import json
import logging
import os
from typing import AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger("gemini.proxy")

UPSTREAM_BASE = os.environ.get(
    "GEMINI_UPSTREAM_BASE_URL",
    "https://webai-to-api-daz.onrender.com/v1/stateless",
).rstrip("/")
UPSTREAM_KEY_ENV = "GEMINI_UPSTREAM_API_KEY"
CLIENT_KEY_ENV = "PERPLEXITY_API_KEY"  # Retained to avoid breaking existing clients.


def _auth_error(request: Request):
    expected = os.environ.get("API_KEY") or os.environ.get(CLIENT_KEY_ENV)
    if not expected:
        return JSONResponse(
            {"error": {"message": "Server API key is not configured.",
                       "type": "server_error", "code": "api_key_not_configured"}},
            status_code=503,
        )
    scheme, sep, supplied = request.headers.get("authorization", "").partition(" ")
    if not sep or scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip(), expected):
        return JSONResponse(
            {"error": {"message": "Invalid or missing API key.",
                       "type": "authentication_error", "code": "invalid_api_key"}},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


def _upstream_headers():
    key = os.environ.get(UPSTREAM_KEY_ENV) or os.environ.get("WEB_API_KEY")
    if not key:
        return None
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _error(message: str, code: str, status: int = 502):
    return JSONResponse(
        {"error": {"message": message, "type": "server_error", "code": code}},
        status_code=status,
    )


async def list_models(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    headers = _upstream_headers()
    if headers is None:
        return _error("Gemini upstream key is not configured.", "upstream_key_missing", 503)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            upstream = await client.get(f"{UPSTREAM_BASE}/models", headers=headers)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
    except httpx.TimeoutException:
        return _error("Gemini model-list request timed out.", "upstream_timeout", 504)
    except httpx.HTTPError as exc:
        logger.warning("Gemini models request failed (%s)", exc.__class__.__name__)
        return _error("Could not reach the Gemini upstream.", "upstream_connection_error")


async def chat_completions(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    headers = _upstream_headers()
    if headers is None:
        return _error("Gemini upstream key is not configured.", "upstream_key_missing", 503)

    try:
        raw = await request.body()
        if len(raw) > 20 * 1024 * 1024:
            return _error("Request body exceeds 20 MB.", "request_too_large", 413)
        payload = json.loads(raw or b"{}")
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "Request body must be a JSON object.",
                           "type": "invalid_request_error", "code": "invalid_json"}},
                status_code=400,
            )

        # Gemini WebAPI does not implement these OpenAI sampling/format controls.
        # Ignore them for client compatibility rather than forwarding unsupported fields.
        for field in ("temperature", "top_p", "top_k", "response_format",
                      "parallel_tool_calls", "tool_choice"):
            payload.pop(field, None)
        payload.setdefault("model", "gemini-auto")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        timeout = httpx.Timeout(180.0, connect=15.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            req = client.build_request(
                "POST",
                f"{UPSTREAM_BASE}/chat/completions",
                content=body,
                headers=headers,
            )
            upstream = await client.send(req, stream=payload.get("stream") is True)

            if upstream.status_code >= 400:
                content = await upstream.aread()
                status = upstream.status_code
                content_type = upstream.headers.get("content-type", "application/json")
                await upstream.aclose()
                return Response(content=content, status_code=status, media_type=content_type)

            if payload.get("stream") is True:
                async def stream_body() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        await client.aclose()

                return StreamingResponse(
                    stream_body(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            content = await upstream.aread()
            status = upstream.status_code
            content_type = upstream.headers.get("content-type", "application/json")
            await upstream.aclose()
            return Response(content=content, status_code=status, media_type=content_type)

    except json.JSONDecodeError:
        return JSONResponse(
            {"error": {"message": "Request body must contain valid JSON.",
                       "type": "invalid_request_error", "code": "invalid_json"}},
            status_code=400,
        )
    except httpx.TimeoutException:
        return _error("Gemini upstream timed out.", "upstream_timeout", 504)
    except httpx.HTTPError as exc:
        logger.warning("Gemini chat request failed (%s)", exc.__class__.__name__)
        return _error("Could not reach the Gemini upstream.", "upstream_connection_error")
    except Exception as exc:
        logger.warning("Request handling failed (%s)", exc.__class__.__name__)
        return _error("The API could not process this request.", "proxy_error")


async def health(request: Request):
    auth = _auth_error(request)
    if auth:
        return auth
    configured = bool(os.environ.get(UPSTREAM_KEY_ENV) or os.environ.get("WEB_API_KEY"))
    return JSONResponse({"status": "ok", "provider": "gemini-webapi",
                         "upstream_configured": configured})


api_app = Starlette(routes=[
    Route("/api/v1/models", list_models, methods=["GET", "HEAD"]),
    Route("/api/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/api/v1/health", health, methods=["GET", "HEAD"]),
])


class DispatchMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/"):
            await api_app(scope, receive, send)
            return
        response = JSONResponse({"error": "Not found."}, status_code=404)
        await response(scope, receive, send)


app = Starlette()
app.add_middleware(DispatchMiddleware)
