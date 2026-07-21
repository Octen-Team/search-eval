"""LLM caller with a provider switch: official Anthropic API or OpenRouter.

Selection (LLM_PROVIDER env var):
  LLM_PROVIDER=anthropic   force the official Anthropic endpoint (needs ANTHROPIC_API_KEY)
  LLM_PROVIDER=openrouter  force OpenRouter (needs OPENROUTER_API_KEY)
  unset                    auto: anthropic when ANTHROPIC_API_KEY is set, otherwise openrouter

This module owns ALL call robustness — callers must not wrap their own retries:
- HTTP backoff (429/5xx/connection errors, Retry-After honored, capped 30s)
- empty-completion and 200-with-error-body handling, provider-agnostic, surfaced as
  retryable RuntimeErrors (not misreported as JSON parse failures)
- attempt-level retry with a short sleep (`attempts` param, default 3)
- JSON parsing with fence/prose/multi-object recovery (parse_llm_json)

Models are given as Anthropic names (e.g. claude-sonnet-4-6) or full provider ids
('openai/gpt-5.2' passes through). Unknown short names raise immediately with a clear
message instead of producing an invalid id that 404s mid-run.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

from .backends import _load_dotenv
from .common import http_session

_load_dotenv()

_RETRY_SLEEP = 1.5  # base backoff between attempt-level retries (tests set this to 0)

_OPENROUTER_IDS = {
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-opus-4-8": "anthropic/claude-opus-4.8",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
}


def _provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p in ("anthropic", "openrouter"):
        return p
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter"


def _openrouter_id(model: str) -> str:
    if "/" in model:
        return model
    if model in _OPENROUTER_IDS:
        return _OPENROUTER_IDS[model]
    m = re.fullmatch(r"(claude-[a-z]+)-(\d+)-(\d+)", model)
    if m:
        return f"anthropic/{m.group(1)}-{m.group(2)}.{m.group(3)}"
    raise ValueError(
        f"cannot map model {model!r} to an OpenRouter id — pass a full provider id "
        f"(e.g. 'openai/gpt-5.2', 'anthropic/claude-sonnet-4.6') or add it to _OPENROUTER_IDS")


def is_retryable(e: Exception) -> bool:
    """Transient failures worth another attempt; auth/validation errors are not."""
    if isinstance(e, requests.exceptions.HTTPError):
        code = e.response.status_code if e.response is not None else None
        return code == 429 or (code is not None and code >= 500)
    return isinstance(e, (RuntimeError, json.JSONDecodeError,
                          requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def _post_with_backoff(url: str, headers: dict, body: dict, timeout: int = 120, attempts: int = 4):
    """POST with exponential backoff on 429/5xx/connection errors; honors Retry-After (capped 30s)."""
    last_exc = None
    for attempt in range(attempts):
        try:
            resp = http_session().post(url, headers=headers, json=body, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt == attempts - 1:
                raise
            time.sleep(min(2.0 * (2 ** attempt), 30))
            continue
        if resp.status_code in (429, 500, 502, 503, 529) and attempt < attempts - 1:
            ra = resp.headers.get("retry-after", "")
            delay = float(ra) if ra.replace(".", "", 1).isdigit() else 2.0 * (2 ** attempt)
            time.sleep(min(delay, 30))
            continue
        resp.raise_for_status()
        return resp
    raise last_exc  # pragma: no cover — loop always returns or raises earlier


def _openrouter_body(model: str, system: str, user: str, max_tokens: int, json_mode: bool,
                     temperature: float | None = None) -> dict:
    body = {"model": _openrouter_id(model), "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if json_mode:
        # forces valid-JSON output on supporting models; unsupported models silently ignore it,
        # so the parse fallbacks in parse_llm_json stay as defense in depth
        body["response_format"] = {"type": "json_object"}
    if temperature is not None:
        body["temperature"] = temperature
    return body


def _fetch_text(system: str, user: str, model: str, max_tokens: int, json_mode: bool,
                temperature: float | None = None) -> str:
    """One raw completion. Raises RuntimeError for body-level provider failures so they are
    retried by the caller loop instead of surfacing as misleading KeyErrors."""
    if _provider() == "anthropic":
        anth_body = {"model": model, "max_tokens": max_tokens, "system": system,
                     "messages": [{"role": "user", "content": user}]}
        if temperature is not None:
            anth_body["temperature"] = temperature
        resp = _post_with_backoff(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body=anth_body,
        )
        return "".join(b.get("text", "") for b in resp.json()["content"])
    resp = _post_with_backoff(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "content-type": "application/json"},
        body=_openrouter_body(model, system, user, max_tokens, json_mode, temperature),
    )
    payload = resp.json()
    if "choices" not in payload:
        # OpenRouter delivers provider failures/moderation as HTTP 200 + {"error": {...}}
        raise RuntimeError(f"openrouter error body: {payload.get('error')}")
    return payload["choices"][0]["message"]["content"] or ""


def call_llm_text(system: str, user: str, model: str, max_tokens: int = 1024,
                  json_mode: bool = False, attempts: int = 3,
                  temperature: float | None = None) -> str:
    """Completion with provider-agnostic retries. Empty completions (provider overload,
    observed 2026-07 in bursts) are retried and, if persistent, raised as RuntimeError —
    never returned as "" to be misdiagnosed as a JSON parse bug downstream."""
    last_err: Exception = RuntimeError("no attempt made")
    for attempt in range(attempts):
        try:
            text = _fetch_text(system, user, model, max_tokens, json_mode, temperature)
            if text.strip():
                return text
            last_err = RuntimeError("empty completion (provider overload?)")
        except Exception as e:  # noqa: BLE001
            if not is_retryable(e):
                raise
            last_err = e
        if attempt < attempts - 1:
            time.sleep(_RETRY_SLEEP * (attempt + 1))
    raise last_err


def call_llm_json(system: str, user: str, model: str, max_tokens: int = 1024,
                  attempts: int = 3, temperature: float | None = None) -> dict:
    """JSON completion; parse failures count as retryable attempts (models are stochastic —
    a re-ask usually fixes malformed output)."""
    last_err: Exception = RuntimeError("no attempt made")
    for attempt in range(attempts):
        try:
            return parse_llm_json(call_llm_text(system, user, model, max_tokens,
                                                json_mode=True, attempts=1,
                                                temperature=temperature))
        except Exception as e:  # noqa: BLE001
            if not is_retryable(e):
                raise
            last_err = e
        if attempt < attempts - 1:
            time.sleep(_RETRY_SLEEP * (attempt + 1))
    raise last_err


def parse_llm_json(text: str) -> dict:
    t = text.strip()
    # strip ONE outer fence pair only, anchored to the whole text — a MULTILINE version of this
    # used to eat code fences INSIDE string values (observed corrupting agent answers, 2026-07)
    t = re.sub(r"^```[a-zA-Z]*[ \t]*\n", "", t)
    t = re.sub(r"\n```[ \t]*$", "", t)
    # strict=False: tolerate literal control characters (raw newlines) inside strings
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        # despite "JSON only" instructions, models occasionally wrap the JSON in prose or emit
        # several concatenated objects; recover the first complete object
        start = t.find("{")
        if start != -1:
            try:
                obj, _ = json.JSONDecoder(strict=False).raw_decode(t[start:])
                return obj
            except json.JSONDecodeError:
                pass
        raise
