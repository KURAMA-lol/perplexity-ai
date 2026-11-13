"""Cookie-authenticated, OpenAI-compatible Perplexity web API. No MCP routes."""
import asyncio
import hmac
import json
import logging
import os
import re
import threading
import time
from uuid import uuid4

os.environ.setdefault("XDG_STATE_HOME", "/tmp/.local/state")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from perplexity.client import Client
from perplexity.config import ENDPOINT_AUTH_SESSION, MODEL_MAPPINGS
from perplexity.exceptions import AuthenticationError, RateLimitError, NetworkError

logger = logging.getLogger("perplexity.api")


def _parse_cookies(raw):
    """Accept JSON object/list, Cookie header string, or Netscape cookie export."""
    if not raw or not raw.strip():
        return {}
    value = raw.strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        if isinstance(parsed.get("cookies"), list):
            parsed = parsed["cookies"]
        elif isinstance(parsed.get("cookies"), dict):
            parsed = parsed["cookies"]
        else:
            return {str(k): str(v) for k, v in parsed.items() if v is not None}

    if isinstance(parsed, list):
        result = {}
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result[str(item[0])] = str(item[1])
            elif isinstance(item, dict) and item.get("name") is not None:
                result[str(item["name"])] = str(item.get("value", ""))
        if result:
            return result

    # Netscape cookie file: domain, flag, path, secure, expiry, name, value.
    if "\t" in value and "\n" in value:
        result = {}
        for line in value.splitlines():
            line = line.strip()
            if not line or (line.startswith("#") and not line.startswith("#HttpOnly_")):
                continue
            if line.startswith("#HttpOnly_"):
                line = line[len("#HttpOnly_"):]
            fields = line.split("\t")
            if len(fields) >= 7:
                result[fields[5]] = fields[6]
        if result:
            return result

    # Browser Cookie header.
    result = {}
    for part in value.split(";"):
        name, sep, cookie_value = part.strip().partition("=")
        if sep and name:
            result[name.strip()] = cookie_value.strip()
    return result


_COOKIE_RAW = os.environ.get("PERPLEXITY_COOKIES", "")
_COOKIES = _parse_cookies(_COOKIE_RAW)
_client = Client(_COOKIES)
_provider_lock = threading.Lock()


def _auth_error(request: Request):
    expected = os.environ.get("PERPLEXITY_API_KEY")
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
        content = _content_text(item.get("content", ""))
        if content:
            lines.append(role.capitalize() + ": " + content)
    if not lines:
        raise ValueError("messages contains no text.")
    return "\n\n".join(lines)


def _extract_answer(result):
    if isinstance(result, dict):
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
        for block in result.get("blocks", []) if isinstance(result.get("blocks"), list) else []:
            md = block.get("markdown_block") if isinstance(block, dict) else None
            if isinstance(md, dict) and isinstance(md.get("answer"), str):
                return md["answer"]
    return ""


def _error(message, error_type="invalid_request_error", code=None, status=400):
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": code}},
        status_code=status,
    )


def _unauthenticated_answer(answer):
    return answer.strip().lower() in {
        "sign up and repeat your request",
        "sign up and repeat your request.",
    }


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
            async def generate():
                previous = ""
                try:
                    with _provider_lock:
                        chunks = await asyncio.to_thread(
                            _client.search, query, mode=mode, model=provider_model,
                            stream=True, require_complete=True
                        )
                        for chunk in chunks:
                            current = _extract_answer(chunk)
                            if not current:
                                continue
                            if _unauthenticated_answer(current):
                                failure = {"error": {"message": "Perplexity rejected the web session. Check that PERPLEXITY_COOKIES contains a currently signed-in perplexity.ai session.",
                                                     "type": "authentication_error",
                                                     "code": "provider_authentication_failed"}}
                                yield "data: " + json.dumps(failure) + "\n\n"
                                yield "data: [DONE]\n\n"
                                return
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
                    logger.warning("Perplexity stream failed (%s)", exc.__class__.__name__)
                    yield "data: " + json.dumps({"error": {"message": "Perplexity request failed.",
                                                           "type": "server_error",
                                                           "code": "provider_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            with _provider_lock:
                result = await asyncio.to_thread(
                    _client.search, query, mode=mode, model=provider_model,
                    stream=False, require_complete=True
                )
            answer = _extract_answer(result)
        except AuthenticationError:
            return _error("Perplexity rejected the web session. Check that PERPLEXITY_COOKIES contains a currently signed-in perplexity.ai session.",
                          "authentication_error", "provider_authentication_failed", 502)
        except RateLimitError:
            return _error("Perplexity rate limit reached; try again later.",
                          "rate_limit_error", "provider_rate_limit", 429)
        except NetworkError as exc:
            logger.warning("Perplexity network error (%s)", exc.__class__.__name__)
            return _error("Perplexity upstream request failed.", "server_error",
                          "provider_network_error", 502)
        except Exception as exc:
            logger.warning("Perplexity request failed (%s)", exc.__class__.__name__)
            return _error("Perplexity request failed.", "server_error", "provider_error", 502)
        if _unauthenticated_answer(answer):
            return _error("Perplexity rejected the web session. Check that PERPLEXITY_COOKIES contains a currently signed-in perplexity.ai session.",
                          "authentication_error", "provider_authentication_failed", 502)
        if not answer:
            return _error("Perplexity returned an empty or incomplete response.",
                          "server_error", "empty_provider_response", 502)
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
    status = {"status": "ok", "provider": "perplexity",
              "cookies_configured": bool(_COOKIES)}
    try:
        with _provider_lock:
            response = await asyncio.to_thread(_client.session.get, ENDPOINT_AUTH_SESSION)
        try:
            data = response.json()
            status["session_authenticated"] = bool(
                isinstance(data, dict) and (data.get("user") or data.get("accessToken"))
            )
            status["session_http_status"] = response.status_code
        except Exception:
            status["session_authenticated"] = False
            status["session_http_status"] = response.status_code
    except Exception as exc:
        logger.warning("Perplexity session check failed (%s)", exc.__class__.__name__)
        status["session_authenticated"] = False
        status["session_check_error"] = True
    return JSONResponse(status)


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
