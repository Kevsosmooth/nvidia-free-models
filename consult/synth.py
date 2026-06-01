"""Consensus synthesis: merge two models' answers into a single actionable verdict.

This is the "council" step -- not a summary of each answer, but a cross-analysis:
where they agree, where they diverge (and who's likely right), and the bottom line.
"""
from . import client

SYNTH_SYSTEM = (
    "You are a neutral synthesizer for a third AI agent. You are given a question and two "
    "independent expert answers (A and B) from different AI models. Produce a tight synthesis "
    "the agent can act on, in exactly these sections:\n"
    "## Agreement\nPoints both make.\n"
    "## Disagreement\nWhere they differ -- and which is more likely correct, with a reason.\n"
    "## Unique\nNotable points only one raised.\n"
    "## Bottom line\nThe single recommended course of action.\n\n"
    "Be concise and decisive. Do NOT just summarize each answer in turn."
)


def synthesize(question, labeled_answers, synth_model="qwen"):
    """labeled_answers: list of (label, text). Returns synthesis text."""
    body = [f"# Question\n{question}\n"]
    for label, text in labeled_answers:
        body.append(f"# Answer {label}\n{text}\n")
    msgs = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": "\n".join(body)},
    ]
    return client.call(synth_model, msgs, temperature=0.3, max_tokens=3000)
