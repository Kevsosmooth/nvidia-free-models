"""Output formatting: human-readable text (default) and JSON (--json)."""
import json

_BAR = "=" * 64


def render_text(mode, question, results, synthesis=None, routing=None):
    """results: list of dicts {label, model, ok, text|error}."""
    out = []
    if routing:
        out.append(f"> [auto] {routing}\n")
    multi = len(results) > 1
    for r in results:
        if multi:
            out.append(f"{_BAR}\n## {r['label']} -- {r['model']}\n{_BAR}")
        out.append(r["text"] if r["ok"] else f"[unavailable] {r['error']}")
        out.append("")
    if synthesis:
        out.append(f"{_BAR}\n## SYNTHESIS\n{_BAR}")
        out.append(synthesis)
    return "\n".join(out).strip()


def render_json(mode, question, results, synthesis=None, routing=None):
    return json.dumps(
        {
            "mode": mode,
            "routing": routing,
            "question": question,
            "answers": [
                {
                    "label": r["label"],
                    "model": r["model"],
                    "ok": r["ok"],
                    "text": r.get("text"),
                    "error": r.get("error"),
                }
                for r in results
            ],
            "synthesis": synthesis,
        },
        indent=2,
    )
