"""Cost tracking у SQLite: лог кожного запиту + агрегації для /usage/*."""
from __future__ import annotations
import pathlib
import sqlite3

from app.config import settings
from app.pricing import chat_cost_usd

_DDL = """
CREATE TABLE IF NOT EXISTS request_costs (
    request_id    TEXT PRIMARY KEY,
    api_key       TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    latency_ms    REAL,
    ttft_ms       REAL,
    cache_hit     INTEGER,
    fallback_used INTEGER,
    created_at    TEXT DEFAULT (datetime('now'))
);
"""


def _conn() -> sqlite3.Connection:
    pathlib.Path(settings.cost_db_path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(settings.cost_db_path)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(_DDL)


def log_request(request_id: str, api_key: str, model: str,
                input_tokens: int, output_tokens: int,
                latency_ms: float, ttft_ms: float,
                cache_hit: bool = False, fallback_used: bool = False) -> float:
    """Записати рядок витрат, повернути вартість запиту."""
    cost = chat_cost_usd(model, input_tokens, output_tokens)
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO request_costs "
            "(request_id, api_key, model, input_tokens, output_tokens, cost_usd, "
            " latency_ms, ttft_ms, cache_hit, fallback_used) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (request_id, api_key, model, input_tokens, output_tokens, cost,
             latency_ms, ttft_ms, int(cache_hit), int(fallback_used)),
        )
    return cost


def get_today(api_key: str) -> dict:
    with _conn() as c:
        r = c.execute(
            "SELECT COUNT(*) reqs, "
            "       COALESCE(SUM(input_tokens + output_tokens), 0) toks, "
            "       COALESCE(SUM(cost_usd), 0) cost "
            "FROM request_costs WHERE api_key = ? AND date(created_at) = date('now')",
            (api_key,),
        ).fetchone()
    return {"requests": r["reqs"], "tokens": r["toks"], "cost_usd": round(r["cost"], 6)}


def get_breakdown(api_key: str) -> dict:
    with _conn() as c:
        by_model = [dict(row) for row in c.execute(
            "SELECT model, COUNT(*) requests, "
            "       SUM(input_tokens + output_tokens) tokens, "
            "       ROUND(SUM(cost_usd), 6) cost_usd "
            "FROM request_costs WHERE api_key = ? GROUP BY model",
            (api_key,),
        ).fetchall()]
        agg = c.execute(
            "SELECT AVG(cache_hit) chr, AVG(fallback_used) fbr, AVG(latency_ms) avg_lat "
            "FROM request_costs WHERE api_key = ?", (api_key,),
        ).fetchone()
        lats = [row["latency_ms"] for row in c.execute(
            "SELECT latency_ms FROM request_costs WHERE api_key = ? ORDER BY latency_ms",
            (api_key,),
        ).fetchall()]
    p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)] if lats else 0.0
    return {
        "by_model": by_model,
        "cache_hit_rate": round(agg["chr"] or 0, 3),
        "fallback_rate": round(agg["fbr"] or 0, 3),
        "avg_latency_ms": round(agg["avg_lat"] or 0, 1),
        "p95_latency_ms": round(p95, 1),
    }