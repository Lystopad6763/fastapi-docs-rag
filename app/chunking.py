"""Structure-aware markdown chunking.

Підхід chunk-by-structure: 1 chunk = 1 логічна секція по заголовках,
з метаданими й розміром у межах 256-1024 токени. Кожен чанк несе breadcrumb заголовків
("Query Parameters > Optional parameters"), який додається на початок тексту —
це дає семантичний контекст і покращує і embedding, і retrieval.

Правила:
  * код у ```...``` НІКОЛИ не ріжеться всередині;
  * секція > MAX_TOKENS ріжеться по безпечних межах (абзац / межа коду);
  * порожні секції пропускаються.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")   # той самий токенайзер, що в text-embedding-3
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{[^}]*\}\s*$")    # прибрати FastAPI "{ #anchor }"
FENCE_RE = re.compile(r"^```")

MAX_TOKENS = 700    # м'який максимум на чанк (типово 256-1024)
MIN_TOKENS = 24     # секції дрібніші — пропускаємо як шум


def ntok(s: str) -> int:
    return len(_ENC.encode(s))


@dataclass
class Chunk:
    source: str      # ім'я файлу (без .md)
    heading: str     # breadcrumb заголовків
    text: str        # breadcrumb + тіло секції (саме це ембедимо)
    tokens: int


def _blocks(body: str) -> list[str]:
    """Розбити тіло секції на блоки: код-фенс = цілий блок, текст — по абзацах."""
    blocks: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in body.split("\n"):
        if FENCE_RE.match(line):
            if not in_code:                       # відкриття коду — злити текст
                if "\n".join(cur).strip():
                    blocks.append("\n".join(cur).rstrip())
                cur = [line]
                in_code = True
            else:                                 # закриття коду — зберегти блок цілим
                cur.append(line)
                blocks.append("\n".join(cur))
                cur = []
                in_code = False
            continue
        if in_code:
            cur.append(line)
        elif line.strip() == "":                  # порожній рядок = межа абзацу
            if "\n".join(cur).strip():
                blocks.append("\n".join(cur).rstrip())
                cur = []
        else:
            cur.append(line)
    if "\n".join(cur).strip():
        blocks.append("\n".join(cur).rstrip())
    return blocks


def _pack(blocks: list[str], budget: int) -> list[str]:
    """Жадібно складати блоки в чанки до budget токенів (код-блок > budget — соло)."""
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


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    stack: list[tuple[int, str]] = []          # ієрархія заголовків (level, title)
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
            flush()                            # тіло належить ПОПЕРЕДНЬОМУ заголовку
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