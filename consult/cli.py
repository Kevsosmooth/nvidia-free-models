"""consult -- a CLI that lets an AI agent (or you) consult NVIDIA's free models.

Modes:
    ask        quick expert answer
    brainstorm options + trade-offs + a recommendation
    review     second-opinion code review of a file or diff
    think      challenge assumptions, find edge cases / failure modes
    code       write code for a task
    consensus  both models with opposing stances, then a synthesis
    models     list every available model id

One model by default (qwen3.5). --both fans out to qwen+kimi concurrently and
synthesizes their answers. --json emits a structured object for programmatic use.

Examples:
    consult ask "what's the cleanest way to debounce in React?"
    consult review -f server.py "focus on auth"
    git diff | consult review
    consult brainstorm --both "ways to cache personalized SSR safely"
    consult consensus "should we adopt tRPC for this project?"
    consult ask --json -m kimi "explain CRDTs in two sentences"
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from . import client, render, router, synth
from .modes import MODES, STANCES, STANCE_LABELS


def build_user_message(prompt_text, files, stdin_text, mode_name):
    parts = []
    if prompt_text:
        parts.append(prompt_text)
    for path in files or []:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as e:
            content = f"<could not read {path}: {e}>"
        parts.append(f"\n--- file: {path} ---\n```\n{content}\n```")
    if stdin_text:
        label = "diff" if mode_name == "review" else "input"
        parts.append(f"\n--- {label} (stdin) ---\n```\n{stdin_text}\n```")
    return "\n".join(parts).strip()


def run_one(model, system, user, temperature, max_tokens, timeout):
    try:
        text = client.call(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        return {"ok": True, "model": client.resolve(model), "text": text}
    except client.ModelUnavailable as e:
        return {"ok": False, "model": client.resolve(model), "error": str(e)}


def parse_args(argv):
    p = argparse.ArgumentParser(prog="consult", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=list(MODES) + ["models"], help="what to do")
    p.add_argument("prompt", nargs="*", help="your question / instruction")
    p.add_argument("-m", "--model", default=client.DEFAULT_MODEL,
                   help=f"model or alias (default: {client.DEFAULT_MODEL}). Aliases: {', '.join(client.MODELS)}")
    p.add_argument("--both", "--compare", action="store_true", dest="both",
                   help="ask both council models (qwen+kimi) and synthesize")
    p.add_argument("--auto", action="store_true",
                   help="let the tool pick the best model(s) for the task and explain why")
    p.add_argument("--stances", action="store_true",
                   help="with --both, steer the two to supportive vs critical (implies --both)")
    p.add_argument("--no-synth", action="store_true", help="with --both, skip the synthesis step")
    p.add_argument("-f", "--file", action="append", help="attach a file's contents (repeatable)")
    p.add_argument("--json", action="store_true", help="emit structured JSON")
    p.add_argument("--max-tokens", type=int, help="override max output tokens")
    p.add_argument("--timeout", type=int, default=180, help="per-call timeout seconds (default 180)")
    p.add_argument("--synth-model", default="qwen", help="model used to synthesize (default qwen)")
    # intermixed so `consult ask -m fast a free text prompt` works (optionals between positionals).
    return p.parse_intermixed_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.mode == "models":
        for mid in client.list_models():
            print(mid)
        return

    mode = MODES[args.mode]
    max_tokens = args.max_tokens or mode.max_tokens

    stdin_text = ""
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()

    prompt_text = " ".join(args.prompt).strip()
    user = build_user_message(prompt_text, args.file, stdin_text, args.mode)
    if not user:
        sys.exit("Nothing to consult about. Provide a prompt, -f file, or piped stdin.")

    routing = None
    if args.auto:
        routed_models, both, routing = router.route(args.mode, prompt_text, bool(args.file))
        both = both or mode.both
        stances = mode.stances
        models = list(client.COUNCIL) if both else routed_models
    else:
        stances = args.stances or mode.stances
        both = args.both or mode.both or stances  # stances implies two models
        models = list(client.COUNCIL) if both else [args.model]

    jobs = []
    for i, mdl in enumerate(models):
        system = mode.system
        label = chr(ord("A") + i) if both else "A"
        if stances and i < len(STANCES):
            system += STANCES[i]
            label = f"{label} ({STANCE_LABELS[i]})"
        jobs.append((mdl, system, label))

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futs = [
            pool.submit(run_one, mdl, system, user, mode.temperature, max_tokens, args.timeout)
            for mdl, system, _ in jobs
        ]
        results = []
        for (mdl, system, label), fut in zip(jobs, futs):
            r = fut.result()
            r["label"] = label
            results.append(r)

    synthesis = None
    ok_results = [r for r in results if r["ok"]]
    if both and not args.no_synth and len(ok_results) >= 2:
        try:
            synthesis = synth.synthesize(
                user, [(r["label"], r["text"]) for r in ok_results], synth_model=args.synth_model
            )
        except client.ModelUnavailable as e:
            synthesis = f"[synthesis unavailable: {e}]"

    renderer = render.render_json if args.json else render.render_text
    print(renderer(args.mode, prompt_text or "(see attached input)", results, synthesis, routing))

    if not ok_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
