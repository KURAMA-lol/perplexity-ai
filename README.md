# Perplexity OpenAI-Compatible API

A small authenticated REST wrapper around the Perplexity web client. Authentication to Perplexity uses the private `PERPLEXITY_COOKIES` environment variable. The public API uses a separate `PERPLEXITY_API_KEY`.

## Endpoints

- `GET /api/v1/models`
- `POST /api/v1/chat/completions`
- `GET /api/v1/health`

## Client

- Base URL: `https://perplexity-ai-dazzmeh-7432.vercel.app/api/v1`
- Authentication: `Authorization: Bearer <PERPLEXITY_API_KEY>`
- Default model: `perplexity`

## Perplexity session

Set `PERPLEXITY_COOKIES` in Vercel Production. Supported formats are a JSON object, JSON list of cookie pairs/objects, a browser Cookie header string, or a Netscape cookie export. Cookie contents are never returned by the API.

## Models

Aliases include `perplexity`/`auto`, `pro`, `reasoning`, and `deep-research`. Specific IDs are listed by `GET /api/v1/models`.

This uses Perplexity's web interface and browser session cookies, not the official Perplexity API. It is unofficial and may break when the website changes. Tool/function calling is not supported by the underlying web client.
