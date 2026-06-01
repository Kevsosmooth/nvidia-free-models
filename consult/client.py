"""Core NVIDIA NIM client: model registry + a robust single completion call.

The free integrate.api.nvidia.com endpoint is flaky (transient connection drops,
occasional 404s on listed-but-undeployed models). This module isolates all of
that: callers get either clean text or a typed ModelUnavailable error.
"""
from __future__ import annotations

import os
import time

import openai
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv()  # repo .env when run from source (python-dotenv walks up from this file)
load_dotenv(find_dotenv(usecwd=True), override=False)  # also a .env in the current dir (pip installs)

# Friendly aliases -> full model ids. Picks come from this project's own testing
# (see CLAUDE.md): qwen3.5 = best-calibrated default, kimi = broad second opinion.
MODELS = {
    "qwen": "qwen/qwen3.5-397b-a17b",
    "kimi": "moonshotai/kimi-k2.6",
    "llama": "meta/llama-3.3-70b-instruct",
    "fast": "meta/llama-3.1-8b-instruct",
    "deepseek": "deepseek-ai/deepseek-v4-flash",
}
DEFAULT_MODEL = "qwen"
# The two-model "council" used by --both / consensus.
COUNCIL = ["qwen", "kimi"]

_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class ModelUnavailable(RuntimeError):
    """A model could not produce an answer (404, or exhausted retries)."""


def resolve(name: str) -> str:
    """Map an alias to a full model id; pass unknown (already-full) ids through."""
    return MODELS.get(name, name)


def _client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise SystemExit("NVIDIA_API_KEY not set (see .env / .env.example).")
    return OpenAI(
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=key,
    )


def call(model, messages, *, max_tokens=4096, temperature=0.3, timeout=180, retries=3):
    """Run one non-streaming completion and return the answer text.

    Retries transient errors with linear backoff. Raises ModelUnavailable on a
    404 (model not deployed) or after exhausting retries. First attempts with
    reasoning/thinking disabled for clean answers, then falls back to a plain
    request if the model rejects that kwarg.
    """
    cli = _client()
    mid = resolve(model)
    got_empty = False
    bad_request = None
    attempt = 0  # global transient-retry budget for the whole call (across both passes)
    # First pass disables reasoning for clean answers; if that yields empty content
    # (some models emit only a reasoning trace) or is rejected, the second pass runs
    # a plain request, which usually populates message.content.
    for extra in ({"chat_template_kwargs": {"thinking": False}}, {}):
        while True:
            try:
                resp = cli.chat.completions.create(
                    model=mid,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra,
                    timeout=timeout,
                    stream=False,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                got_empty = True
                break  # empty content -> try the next pass (e.g. reasoning on)
            except openai.BadRequestError as e:
                bad_request = e  # maybe the extra_body kwarg; try a plain request, but keep the error
                break
            except openai.NotFoundError as e:
                raise ModelUnavailable(f"{mid}: not available on this endpoint (404).") from e
            except _RETRYABLE as e:
                attempt += 1
                if attempt > retries:
                    raise ModelUnavailable(f"{mid}: failed after {retries} retries ({type(e).__name__}).") from e
                time.sleep(min(2 * attempt, 8))
    # Both passes exhausted. Prefer the concrete rejection message over a generic one.
    if bad_request is not None:
        raise ModelUnavailable(f"{mid}: rejected the request ({bad_request}).") from bad_request
    if got_empty:
        raise ModelUnavailable(f"{mid}: returned an empty response.")
    raise ModelUnavailable(f"{mid}: request could not be completed.")


def list_models():
    return sorted(m.id for m in _client().models.list().data)
