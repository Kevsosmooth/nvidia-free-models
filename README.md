# NVIDIA NIM Models — Testing Playground

Hosted models served through NVIDIA's OpenAI-compatible API at
`https://integrate.api.nvidia.com/v1`. The API key is free-tier and works with
any standard OpenAI SDK by just swapping the `base_url`.

---

## Start here (step by step)

### 1. Get your free NVIDIA API key

1. Go to **https://build.nvidia.com** and click **Sign In / Sign Up** (top right).
   Create a free NVIDIA account (any email works; no credit card).
2. **Verify your phone number** when asked — NVIDIA requires a one-time SMS code
   to activate the free tier. Enter your number, type in the code they text you.
3. Go to **https://build.nvidia.com/settings/api-keys** and click
   **Generate API Key**. Copy the key — it starts with `nvapi-`.
   *(Keep it secret, like a password. You can always generate a new one here.)*

What "free" gets you: ~1,000 inference credits to start and ~40 requests/minute —
plenty for testing and personal projects.

### 2. Install this project

You need **git** and **Python 3.9+** installed. Then, in a terminal:

```bash
git clone https://github.com/Kevsosmooth/nvidia-free-models.git
cd nvidia-free-models
bash setup.sh
```

`setup.sh` does everything for you: builds the environment, installs the
libraries, asks you to paste the API key from step 1 (and saves it safely), and
turns `consult` and `nimchat` into commands you can run from any folder.

### 3. Try it

```bash
consult ask "what can you help me with?"
nimchat "tell me a joke"
consult models            # list every available model
```

That's it. The rest of this README is the detailed reference.

<details>
<summary>Prefer to install by hand (no setup.sh)?</summary>

```bash
python3 -m venv .venv                       # make an isolated environment
./.venv/bin/pip install -r requirements.txt # install openai + python-dotenv
cp .env.example .env                        # then edit .env and paste your nvapi-... key
ln -sf "$PWD/bin/consult" ~/.local/bin/consult   # optional: make `consult` global
ln -sf "$PWD/bin/nimchat" ~/.local/bin/nimchat   # optional: make `nimchat` global
```
Make sure `~/.local/bin` is on your `PATH`. Without the symlinks, run the tools as
`./bin/consult ...` / `python chat.py ...`.
</details>

---

## Run

`chat.py` is the simple streaming tester. It's linked onto PATH as `nimchat`
(named to avoid the system `chat` command), so you can run it from anywhere:

```bash
nimchat "Tell me a joke"                       # default model (qwen3.5)
nimchat -m kimi "Hi"                           # short alias: qwen kimi llama fast deepseek
nimchat -m meta/llama-3.3-70b-instruct "Hi"    # or any full model id
nimchat --no-think "2+2?"                      # turn off reasoning trace
nimchat --list                                 # print the full live model list
# (or call it directly: python chat.py "...")
```

To (re)create the link: `ln -sf "$PWD/bin/nimchat" ~/.local/bin/nimchat`.

The script streams output token-by-token. `--no-think` sends
`extra_body={"chat_template_kwargs":{"thinking":False}}` — only meaningful on
reasoning models (DeepSeek V4, Nemotron, Qwen3, GLM, etc.).

---

## `consult` — the agent-to-agent CLI

`consult` is the main tool: a small CLI that lets **an AI agent (like Claude
Code) or you** consult NVIDIA's free models for a second opinion. Think of it as a
free `codex exec` — one model by default, or fan out to two and synthesize.

Inspired by zen-mcp-server's `chat`/`thinkdeep`/`consensus` tools and the
multi-LLM "council" pattern, but a fast CLI (no MCP overhead) and built to ride
out the flaky free tier (retries, graceful per-model failure).

The launcher lives at `bin/consult` and is symlinked onto `PATH`
(`~/.local/bin/consult`), so `consult` runs from any directory. To re-create the
link: `ln -sf "$PWD/bin/consult" ~/.local/bin/consult`. It uses the project venv
automatically — no activation needed.

```bash
consult ask "cleanest way to debounce in React?"          # quick expert answer
consult review -f server.py "focus on auth"               # second-opinion code review
git diff | consult review                                 # review a diff from stdin
consult brainstorm "ways to cache personalized SSR"       # options + tradeoffs + a pick
consult think "is event sourcing right for our app?"      # challenge assumptions / edge cases
consult code "rate limiter in Go, token bucket"           # write code for a task
consult ask --both "best way to dedupe queue messages?"   # qwen + kimi, then a synthesis
consult consensus "should we adopt tRPC here?"            # both, opposing stances, then verdict
consult ask --json -m kimi "explain CRDTs in 2 sentences" # structured output for programs
consult ask --auto "is this O(n log n)? ..."              # tool picks the model + says why
consult models                                            # list all 118 model ids
```

**Flags:** `-m/--model` (alias or full id; aliases: `qwen kimi llama fast deepseek`),
`--auto` (pick the best model(s) for the task automatically and print the reason),
`--both`/`--compare` (run qwen+kimi concurrently, then synthesize), `--stances`
(supportive vs critical), `--no-synth`, `-f/--file` (repeatable), `--json`,
`--max-tokens`, `--timeout`, `--synth-model`.

`--auto` routing (rules from this project's benchmarks): short factual ask → `fast`;
complex/technical ask, or `code`/`review` → `qwen`; `brainstorm`/`think`/`consensus`,
or `code`/`review` on big input → both models + synthesis.

**Defaults:** one model = `qwen` (qwen3.5-397b, our best-calibrated pick);
`--both` = `qwen` + `kimi`. See `consult/` for the code (`client.py` engine,
`modes.py` prompts, `synth.py` council synthesis, `cli.py` orchestration).

### How another agent calls it
The output is plain markdown (or `--json`) printed to stdout — an agent shells out,
reads the result, and folds it into its own reasoning. Example for Claude Code:
```
Bash: consult review -f path/to/file.py --json
```

---

## Available models (118 total, live list as of 2026-06-01)

### Text chat / general LLMs
These are what you'll use most. All accept the standard `chat.completions` call.

| Model ID | Notes |
|----------|-------|
| `deepseek-ai/deepseek-v4-pro` | DeepSeek's flagship reasoning model (your default) |
| `deepseek-ai/deepseek-v4-flash` | Faster, cheaper DeepSeek variant |
| `meta/llama-4-maverick-17b-128e-instruct` | Llama 4, MoE, long context |
| `meta/llama-3.3-70b-instruct` | Strong all-rounder, very capable |
| `meta/llama-3.1-70b-instruct` | Previous-gen 70B |
| `meta/llama-3.1-8b-instruct` | Small, fast |
| `meta/llama-3.2-3b-instruct` / `meta/llama-3.2-1b-instruct` | Tiny, low-latency |
| `meta/llama2-70b` | Legacy |
| `qwen/qwen3.5-397b-a17b` | Qwen 3.5 flagship MoE |
| `qwen/qwen3.5-122b-a10b` | Qwen 3.5 mid MoE |
| `qwen/qwen3-next-80b-a3b-instruct` | Efficient Qwen MoE |
| `moonshotai/kimi-k2.6` | Moonshot Kimi, very long context |
| `z-ai/glm-5.1` | Zhipu GLM 5.1 |
| `minimaxai/minimax-m2.7` | MiniMax |
| `stepfun-ai/step-3.7-flash` / `stepfun-ai/step-3.5-flash` | StepFun fast models |
| `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | OpenAI open-weight models |
| `mistralai/mistral-large-3-675b-instruct-2512` | Mistral Large 3 (largest) |
| `mistralai/mistral-medium-3.5-128b` | Mistral Medium 3.5 |
| `mistralai/mistral-small-4-119b-2603` | Mistral Small 4 |
| `mistralai/ministral-14b-instruct-2512` | Compact Mistral |
| `mistralai/mistral-large-2-instruct` / `mistralai/mistral-large` | Older Mistral Large |
| `mistralai/mistral-nemotron` | Mistral × NVIDIA |
| `mistralai/mistral-7b-instruct-v0.3` | Small classic |
| `mistralai/mixtral-8x22b-v0.1` / `mistralai/mixtral-8x7b-instruct-v0.1` | Mixtral MoE |
| `nv-mistralai/mistral-nemo-12b-instruct` | Mistral NeMo 12B |
| `google/gemma-4-31b-it` | Gemma 4 |
| `google/gemma-3-12b-it` / `google/gemma-3-4b-it` | Gemma 3 |
| `google/gemma-3n-e4b-it` / `google/gemma-3n-e2b-it` | Gemma 3n (edge) |
| `google/gemma-2-2b-it` / `google/gemma-2b` / `google/recurrentgemma-2b` | Gemma 2 / tiny |
| `microsoft/phi-4-mini-instruct` | Phi-4 mini |
| `microsoft/phi-3.5-moe-instruct` | Phi-3.5 MoE |
| `ibm/granite-3.0-8b-instruct` / `ibm/granite-3.0-3b-a800m-instruct` | IBM Granite |
| `databricks/dbrx-instruct` | Databricks DBRX |
| `ai21labs/jamba-1.5-large-instruct` | AI21 Jamba (SSM hybrid) |
| `01-ai/yi-large` | Yi Large |
| `abacusai/dracarys-llama-3.1-70b-instruct` | Coding-tuned Llama |
| `bytedance/seed-oss-36b-instruct` | ByteDance Seed |
| `upstage/solar-10.7b-instruct` | Upstage Solar |
| `zyphra/zamba2-7b-instruct` | Zamba2 (SSM hybrid) |
| `aisingapore/sea-lion-7b-instruct` | SE-Asian languages |
| `sarvamai/sarvam-m` | Indic languages |
| `stockmark/stockmark-2-100b-instruct` | Japanese |
| `writer/palmyra-creative-122b` | Creative writing |

### Reasoning (NVIDIA Nemotron family)
| Model ID | Notes |
|----------|-------|
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Largest Nemotron |
| `nvidia/nemotron-3-super-120b-a12b` | Nemotron 3 Super |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` / `...-v1` | Super 49B |
| `nvidia/llama-3.1-nemotron-70b-instruct` / `...-51b-instruct` | Nemotron 70B/51B |
| `nvidia/nemotron-3-nano-30b-a3b` / `nvidia/nemotron-nano-3-30b-a3b` | Nemotron 3 Nano |
| `nvidia/nvidia-nemotron-nano-9b-v2` | Nano 9B |
| `nvidia/llama-3.1-nemotron-nano-8b-v1` | Nano 8B |
| `nvidia/mistral-nemo-minitron-8b-8k-instruct` | Minitron |
| `nvidia/nemotron-mini-4b-instruct` | Mini 4B |
| `nvidia/nemotron-4-340b-instruct` | Nemotron 4 340B |
| `nvidia/ising-calibration-1-35b-a3b` | Experimental |
| `nvidia/llama3-chatqa-1.5-70b` | RAG / QA tuned |

### Code generation
| Model ID | Notes |
|----------|-------|
| `qwen/qwen3-coder-480b-a35b-instruct` | Largest coder, MoE |
| `mistralai/codestral-22b-instruct-v0.1` | Codestral |
| `meta/codellama-70b` | Code Llama |
| `bigcode/starcoder2-15b` | StarCoder2 |
| `ibm/granite-34b-code-instruct` / `ibm/granite-8b-code-instruct` | Granite Code |
| `google/codegemma-7b` / `google/codegemma-1.1-7b` | CodeGemma |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | DeepSeek Coder |

### Vision / multimodal (text + image input)
| Model ID | Notes |
|----------|-------|
| `meta/llama-3.2-90b-vision-instruct` / `meta/llama-3.2-11b-vision-instruct` | Llama Vision |
| `microsoft/phi-4-multimodal-instruct` | Phi-4 multimodal (audio+image) |
| `microsoft/phi-3-vision-128k-instruct` | Phi-3 Vision |
| `microsoft/kosmos-2` | Grounding / region |
| `nvidia/nemotron-nano-12b-v2-vl` | Nemotron vision |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | Nemotron nano vision |
| `nvidia/cosmos-reason2-8b` | Physical-world reasoning |
| `nvidia/vila` / `nvidia/neva-22b` | VILA / NeVA |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Omni (multimodal reasoning) |
| `adept/fuyu-8b` | Fuyu |
| `google/deplot` | Chart/plot → text |
| `nvidia/nemoretriever-parse` / `nvidia/nemotron-parse` | Document parsing |

### Embeddings / retrieval (not chat — use the embeddings endpoint)
`baai/bge-m3`, `snowflake/arctic-embed-l`, `nvidia/nv-embed-v1`,
`nvidia/nv-embedqa-e5-v5`, `nvidia/nv-embedqa-mistral-7b-v2`, `nvidia/embed-qa-4`,
`nvidia/llama-3.2-nv-embedqa-1b-v1`, `nvidia/llama-nemotron-embed-1b-v2`,
`nvidia/llama-nemotron-embed-vl-1b-v2`, `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1`,
`nvidia/nv-embedcode-7b-v1` (code), `nvidia/nvclip` (image)

### Safety / guard / moderation
`meta/llama-guard-4-12b`, `nvidia/llama-3.1-nemoguard-8b-content-safety`,
`nvidia/llama-3.1-nemoguard-8b-topic-control`, `nvidia/llama-3.1-nemotron-safety-guard-8b-v3`,
`nvidia/nemotron-3-content-safety`, `nvidia/nemotron-content-safety-reasoning-4b`,
`nvidia/gliner-pii` (PII detection), `nvidia/ai-synthetic-video-detector`

### Domain-specific
| Model ID | Domain |
|----------|--------|
| `writer/palmyra-med-70b` / `writer/palmyra-med-70b-32k` | Medical |
| `writer/palmyra-fin-70b-32k` | Finance |
| `nvidia/riva-translate-4b-instruct` / `...-v1.1` | Translation |
| `nvidia/nemotron-4-340b-reward` | Reward model (RLHF scoring) |

---

## Notes
- This is the **OpenAI-compatible** API, so the same code points at any model by
  changing the `model=` string. Embeddings and reranking use different endpoints.
- Some reasoning models emit a thinking trace; pass `--no-think` to suppress it.
- The live list can change — `python chat.py --list` always reprints the current set.
- Your key lives in `.env` (gitignored, chmod 600). Never commit it.
