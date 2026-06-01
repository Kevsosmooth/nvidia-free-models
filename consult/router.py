"""Rule-based model routing: pick the best model(s) for a task, transparently.

Rules are derived from this project's own benchmarking (see CLAUDE.md): qwen3.5 is
the best-calibrated single model; the qwen+kimi council adds diversity for open-ended
work; the small fast model is plenty for short factual asks. The router returns a
plain-English reason so the choice is never a black box.
"""
import re

_HARD_HINTS = (
    "architecture", "security", "concurren", "distributed", "algorithm", "debug",
    "design", "trade-off", "tradeoff", "compare", "prove", "complexity", "race condition",
    "scalab", "optimi", "why does", "how does",
)
_CODE_RE = re.compile(r"```|(\bdef )|(\bclass )|(\bfunction\b)|=>|(\bimport )|(\bSELECT )", re.I)


def route(mode_name, prompt_text, has_files):
    """Decide which model(s) to use.

    Returns (models, both, reason):
      models -- list of aliases when running a single/explicit set, or None when
                `both` is True (the caller uses the COUNCIL).
      both   -- True to fan out to the two-model council and synthesize.
      reason -- one-line explanation for the user/agent.
    """
    text = prompt_text or ""
    n = len(text)
    looks_like_code = has_files or bool(_CODE_RE.search(text))
    hard = looks_like_code or n > 400 or any(h in text.lower() for h in _HARD_HINTS)

    if mode_name == "consensus":
        return (None, True, "consensus -> both models, opposing stances (decision question)")
    if mode_name == "brainstorm":
        return (None, True, "brainstorm -> both models for idea diversity")
    if mode_name == "think":
        return (None, True, "think -> both models for diverse failure-mode coverage")
    if mode_name in ("code", "review"):
        if has_files or n > 800:
            return (None, True, f"{mode_name} on substantial input -> both models to cross-check")
        return (["qwen"], False, f"{mode_name} -> qwen (best-calibrated for correctness)")
    # ask
    if hard:
        return (["qwen"], False, "complex/technical question -> qwen (deep reasoning)")
    return (["fast"], False, "short factual question -> fast model (llama-3.1-8b) for speed")
