"""Structure-aware markdown chunking.

Chunk-by-structure approach: one chunk = one logical section delimited by headings,
carrying metadata and a size in the 256-1024 token range. Each chunk is prefixed with a
heading breadcrumb ("Query Parameters > Optional parameters"); this adds semantic context
and improves both embedding and retrieval.

Rules:
  * code fences (```...```) are NEVER split internally;
  * a section larger than MAX_TOKENS is split on safe boundaries (paragraph / code fence);
  * empty sections are skipped.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")   # same tokenizer as text-embedding-3
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{[^}]*\}\s*$")    # strip FastAPI's "{ #anchor }" suffixes
FENCE_RE = re.compile(r"^```")

MAX_TOKENS = 700    # soft maximum per chunk (typically 256-1024)
MIN_TOKENS = 24     # smaller sections are skipped as noise


def ntok(s: str) -> int:
    return len(_ENC.encode(s))


@dataclass
class Chunk:
    source: str      # file name (without .md)
    heading: str     # heading breadcrumb
    text: str        # breadcrumb + section body (this is what gets embedded)
    tokens: int


def _blocks(body: str) -> list[str]:
    """Split a section body into blocks: a code fence is one whole block, text is split by paragraph."""
    blocks: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in body.split("\n"):
        if FENCE_RE.match(line):
            if not in_code:                       # opening fence — flush accumulated text
                if "\n".join(cur).strip():
                    blocks.append("\n".join(cur).rstrip())
                cur = [line]
                in_code = True
            else:                                 # closing fence — keep the code block intact
                cur.append(line)
                blocks.append("\n".join(cur))
                cur = []
                in_code = False
            continue
        if in_code:
            cur.append(line)
        elif line.strip() == "":                  # blank line = paragraph boundary
            if "\n".join(cur).strip():
                blocks.append("\n".join(cur).rstrip())
                cur = []
        else:
            cur.append(line)
    if "\n".join(cur).strip():
        blocks.append("\n".join(cur).rstrip())
    return blocks


def _pack(blocks: list[str], budget: int) -> list[str]:
    """Greedily pack blocks into chunks up to `budget` tokens (a code block larger than budget stands alone)."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for b in blocks:
        bt = ntok(b)
        if cur and cur_tok + bt > budget:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(b)
        cur_tok += bt
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def prose_only(text: str) -> str:
    """Return the text WITHOUT code blocks (keeping the breadcrumb + prose).

    Used to test whether it is better NOT to embed code (which dilutes the dense vector),
    keeping it only in the payload. The breadcrumb `# Heading` is preserved (it is not a code
    fence), so even a code-only chunk does not become empty.
    """
    out: list[str] = []
    in_code = False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_code = not in_code
            continue                       # drop the ``` fence lines themselves too
        if not in_code:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    stack: list[tuple[int, str]] = []          # heading hierarchy (level, title)
    sections: list[tuple[str, str]] = []       # (breadcrumb, body)
    body: list[str] = []
    in_code = False

    def crumb() -> str:
        return " > ".join(t for _, t in stack)

    def flush():
        if body and "\n".join(body).strip():
            sections.append((crumb(), "\n".join(body).strip()))
        body.clear()

    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_code = not in_code
            body.append(line)
            continue
        m = None if in_code else HEADER_RE.match(line)
        if m:
            flush()                            # the body belongs to the PREVIOUS heading
            level = len(m.group(1))
            title = ANCHOR_RE.sub("", m.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()

    chunks: list[Chunk] = []
    for head, bodytext in sections:
        prefix = f"# {head}\n\n" if head else ""
        if ntok(prefix + bodytext) <= MAX_TOKENS:
            parts = [bodytext]
        else:
            parts = _pack(_blocks(bodytext), MAX_TOKENS - ntok(prefix))
        n = len(parts)
        for i, part in enumerate(parts):
            label = head + (f" (part {i + 1}/{n})" if n > 1 else "")
            ptext = (f"# {label}\n\n" if head else "") + part
            t = ntok(ptext)
            if t >= MIN_TOKENS:
                chunks.append(Chunk(source=source, heading=label, text=ptext, tokens=t))
    return chunks