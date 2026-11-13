# Gemini OpenAI-Compatible API

A small authenticated proxy that exposes the existing Gemini WebAPI service through an OpenAI-compatible API. No Perplexity client, Perplexity cookies, or MCP server is included.

## Endpoints

- `GET /api/v1/models`
- `POST /api/v1/chat/completions`
- `GET /api/v1/health`

## Client configuration

- Base URL: `https://perplexity-ai-dazzmeh-7432.vercel.app/api/v1`
- Chat completions: `/chat/completions`
- Models: `/models`
- Authentication: `Authorization: Bearer <API_KEY>`
- Default model: `gemini-auto`

Example request:

```json
{
  "model": "gemini-auto",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}
```

The proxy forwards chat requests, streaming, and tool definitions to the configured Gemini WebAPI stateless endpoint. OpenAI sampling/format controls unsupported by Gemini WebAPI are ignored for client compatibility.

## Vercel environment variables

- `PERPLEXITY_API_KEY`: existing client-facing Bearer key (kept to avoid breaking clients).
- `GEMINI_UPSTREAM_BASE_URL`: upstream base URL; defaults to the Daz WebAI-to-API stateless endpoint.
- `GEMINI_UPSTREAM_API_KEY`: private upstream service key. Never put it in client apps or source control.

The upstream key is sent only from the Vercel server to the Gemini service.
