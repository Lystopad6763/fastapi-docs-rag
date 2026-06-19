"""Build the documentation corpus.

Downloads every .md file from the FastAPI docs/en/docs/tutorial/ directory and
makes each one self-contained:
  1) resolves include directives `{* ../../docs_src/....py *}` by inlining the
     referenced source code;
  2) flattens MkDocs admonitions (`/// tip ... ///`) into plain text;
  3) writes the result to data/docs/ (one .md file per tutorial section).

Run:  python scripts/fetch_docs.py
"""
from __future__ import annotations
import json
import pathlib
import re
import urllib.request

REPO = "fastapi/fastapi"
BRANCH = "master"
DOC_PREFIX = "docs/en/docs/tutorial/"     # directory to download (recursively)
DOC_ROOT = "docs/en/docs/"
OUT_DIR = pathlib.Path("data/docs")
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"

INCLUDE_RE = re.compile(r"\{\*\s*([^*}]+?\.py)(?:\s+[^*}]*?)?\*\}")
ADMON_OPEN = re.compile(r"^///\s*(\w+)\s*(?:\|\s*(.*?))?\s*$")
ADMON_CLOSE = re.compile(r"^///\s*$")
ADMON_KINDS = {"tip", "note", "warning", "info", "danger", "check", "question", "example", "abstract"}


def http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "fetch-docs"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")


def resolve_includes(md: str, cache: dict[str, str | None]) -> tuple[str, int, int]:
    ok = fail = 0

    def repl(m: re.Match) -> str:
        nonlocal ok, fail
        path = m.group(1).strip()
        i = path.find("docs_src/")              # robust against ../../ prefixes
        rel = path[i:] if i >= 0 else path.lstrip("./")
        if rel not in cache:
            try:
                cache[rel] = http_get(RAW + rel)
            except Exception:
                cache[rel] = None
        code = cache[rel]
        if code is None:
            fail += 1
            return f"\n```python\n# (code example {rel} unavailable)\n```\n"
        ok += 1
        return f"\n```python\n{code.rstrip()}\n```\n"

    return INCLUDE_RE.sub(repl, md), ok, fail


def clean_admonitions(md: str) -> str:
    out, depth = [], 0
    for line in md.splitlines():
        s = line.strip()
        mo = ADMON_OPEN.match(s)
        if mo and mo.group(1).lower() in ADMON_KINDS:
            label = mo.group(1).capitalize()
            title = (mo.group(2) or "").strip()
            out.append(f"\n**{label}{': ' + title if title else ''}**")
            depth += 1
            continue
        if ADMON_CLOSE.match(s) and depth > 0:
            depth -= 1
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.md"):
        old.unlink()

    tree = json.loads(http_get(TREE_API))["tree"]
    md_paths = sorted(
        n["path"] for n in tree
        if n["type"] == "blob" and n["path"].startswith(DOC_PREFIX) and n["path"].endswith(".md")
    )
    print(f"Found {len(md_paths)} markdown files in {DOC_PREFIX}")

    code_cache: dict[str, str | None] = {}
    inc_ok = inc_fail = 0
    for p in md_paths:
        md = http_get(RAW + p)
        md, ok, fail = resolve_includes(md, code_cache)
        md = clean_admonitions(md)
        flat = p[len(DOC_ROOT):].replace("/", "__")
        (OUT_DIR / flat).write_text(md, encoding="utf-8")
        inc_ok += ok
        inc_fail += fail
    print(f"Inlined code includes: {inc_ok} ok, {inc_fail} fail ({len(code_cache)} unique .py)")
    print(f"Wrote {len(md_paths)} self-contained files -> {OUT_DIR}/")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total = sum(len(enc.encode((OUT_DIR / f.name).read_text(encoding="utf-8")))
                    for f in OUT_DIR.glob("*.md"))
        print(f"Corpus: {total:,} tokens")
    except Exception as e:  # noqa: BLE001
        print("tiktoken count skipped:", e)


if __name__ == "__main__":
    main()