#!/usr/bin/env python3
"""Extract each model's compare_versions() and run it against a SemVer 2.0.0 suite."""
import re
import sys
import traceback

FILES = {
    "qwen3-coder-480b": "/tmp/c_qwencoder.txt",
    "codestral-22b": "/tmp/c_codestral.txt",
    "qwen3.5-397b": "/tmp/c_qwen35.txt",
    "kimi-k2.6": "/tmp/c_kimi.txt",
}

# Official semver.org precedence chain (strictly increasing).
CHAIN = [
    "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
    "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0",
]

# (a, b, expected) direct assertions.
CASES = [
    ("1.0.0", "1.0.0", 0),
    ("2.0.0", "1.0.0", 1),
    ("1.0.0", "2.0.0", -1),
    ("1.2.0", "1.1.9", 1),
    ("1.0.10", "1.0.2", 1),          # numeric, not string, on patch
    ("1.0.0-alpha", "1.0.0", -1),    # pre-release < release
    ("1.0.0-alpha.2", "1.0.0-alpha.11", -1),  # numeric identifiers numerically
    ("1.0.0-alpha.1", "1.0.0-alpha.beta", -1),  # numeric < alphanumeric
    ("1.0.0-alpha", "1.0.0-alpha.1", -1),       # fewer fields < more fields
    ("1.0.0+build.1", "1.0.0+build.2", 0),       # build metadata ignored
    ("1.0.0+x", "1.0.0", 0),                      # build metadata ignored
    ("1.0.0-beta.11", "1.0.0-beta.2", 1),
]


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: maybe no fence, grab from first 'def compare_versions'
    if "def compare_versions" in text:
        return text[text.index("def compare_versions"):]
    return None


def norm(v):
    """Map any return value to -1/0/1 sign for lenient grading."""
    try:
        return (v > 0) - (v < 0)
    except TypeError:
        return v


def grade(name: str, path: str) -> None:
    try:
        with open(path) as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"{name:18} FILE MISSING")
        return

    code = extract_code(raw)
    if not code:
        print(f"{name:18} NO CODE BLOCK FOUND")
        return

    ns: dict = {}
    try:
        exec(code, ns)
    except Exception:
        print(f"{name:18} EXEC ERROR: {traceback.format_exc().splitlines()[-1]}")
        return

    fn = ns.get("compare_versions")
    if not callable(fn):
        print(f"{name:18} no compare_versions() defined")
        return

    passed = failed = errored = 0
    fails = []

    for a, b, exp in CASES:
        try:
            got = norm(fn(a, b))
            if got == exp:
                passed += 1
            else:
                failed += 1
                fails.append(f"  compare({a!r},{b!r})={got} expected {exp}")
        except Exception as e:
            errored += 1
            fails.append(f"  compare({a!r},{b!r}) raised {type(e).__name__}: {e}")

    # full chain: every i<j must be -1, and reverse 1
    chain_fail = 0
    for i in range(len(CHAIN)):
        for j in range(len(CHAIN)):
            exp = (i > j) - (i < j)
            try:
                if norm(fn(CHAIN[i], CHAIN[j])) != exp:
                    chain_fail += 1
            except Exception:
                chain_fail += 1

    total_cases = len(CASES)
    chain_total = len(CHAIN) ** 2
    print(f"{name:18} cases {passed}/{total_cases}  | chain {chain_total - chain_fail}/{chain_total}"
          f"  | errors {errored}")
    for line in fails[:6]:
        print(line)


if __name__ == "__main__":
    for name, path in FILES.items():
        grade(name, path)
