#!/usr/bin/env python3
"""Quick streaming tester for NVIDIA NIM hosted models.

Usage (or run it as `nimchat ...` once linked onto PATH):
    nimchat "Explain quantum entanglement"          # default model (qwen)
    nimchat -m kimi "write a haiku"                  # short alias
    nimchat -m meta/llama-3.3-70b-instruct "hi"      # or any full model id
    nimchat --list                                   # print all model IDs
    echo "hi" | nimchat                              # read prompt from stdin
"""
import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv()  # repo .env when run from source
load_dotenv(find_dotenv(usecwd=True), override=False)  # also a .env in the current dir (pip installs)

# Short aliases for the handful you'll actually type (mirrors consult).
ALIASES = {
    "qwen": "qwen/qwen3.5-397b-a17b",
    "kimi": "moonshotai/kimi-k2.6",
    "llama": "meta/llama-3.3-70b-instruct",
    "fast": "meta/llama-3.1-8b-instruct",
    "deepseek": "deepseek-ai/deepseek-v4-flash",
}
DEFAULT_MODEL = "qwen"  # deepseek-v4-pro does not stream on the free tier; qwen is the tested best


def make_client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY not set. Put it in .env (see .env.example).")
    return OpenAI(
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=key,
    )


def list_models(client: OpenAI) -> None:
    for m in sorted(client.models.list().data, key=lambda x: x.id):
        print(m.id)


def chat(client: OpenAI, model: str, prompt: str, no_think: bool) -> None:
    extra = {"chat_template_kwargs": {"thinking": False}} if no_think else {}
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body=extra,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test NVIDIA NIM hosted models.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model id (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="list every available model id and exit")
    parser.add_argument("--no-think", action="store_true", help="disable reasoning/thinking output where supported")
    parser.add_argument("prompt", nargs="*", help="prompt text (or pipe via stdin)")
    args = parser.parse_args()

    client = make_client()

    if args.list:
        list_models(client)
        return

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        prompt = input("Prompt: ").strip()
    if not prompt:
        sys.exit("No prompt given.")

    model = ALIASES.get(args.model, args.model)
    print(f"--- {model} ---")
    chat(client, model, prompt, args.no_think)


if __name__ == "__main__":
    main()
