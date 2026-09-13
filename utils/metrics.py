"""SQLite-backed utilisation metrics for the PAL MCP server.

Records one row per tool call and one row per model completion, so that
"how much are the server and the models being used" can be answered after
the fact. Writes are best-effort: a metrics failure must never break a
tool call.

Environment:
    PAL_METRICS_ENABLED  "false" to disable recording entirely (default on)
    PAL_METRICS_DB       override the database path (default logs/metrics.db)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "record_model_usage",
    "track_tool_call",
    "metrics_enabled",
    "get_db_path",
    "connect",
    "estimate_cost",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    tool          TEXT    NOT NULL,
    client        TEXT,
    continuation  INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    status        TEXT    NOT NULL,
    error_type    TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_ts   ON calls(ts);
CREATE INDEX IF NOT EXISTS idx_calls_tool ON calls(tool);

CREATE TABLE IF NOT EXISTS model_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id       INTEGER REFERENCES calls(id),
    ts            REAL    NOT NULL,
    tool          TEXT,
    model         TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL
);
CREATE INDEX IF NOT EXISTS idx_model_ts    ON model_calls(ts);
CREATE INDEX IF NOT EXISTS idx_model_model ON model_calls(model);
"""

_init_lock = threading.Lock()
_initialised = False


def metrics_enabled() -> bool:
    """Metrics recording is on unless explicitly disabled."""
    return os.getenv("PAL_METRICS_ENABLED", "true").strip().lower() not in ("false", "0", "no")


def get_db_path() -> Path:
    override = os.getenv("PAL_METRICS_DB")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "logs" / "metrics.db"


def connect(readonly: bool = False) -> sqlite3.Connection:
    """Open the metrics database, creating the schema on first use."""
    global _initialised
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    if not readonly:
        with _init_lock:
            if not _initialised:
                # WAL keeps concurrent readers (the report script, a dashboard)
                # from blocking the server's writes.
                with contextlib.suppress(sqlite3.DatabaseError):
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                conn.commit()
                _initialised = True
    return conn


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

_PRICING_CACHE: dict[str, dict[str, float]] | None = None


def _pricing() -> dict[str, dict[str, float]]:
    """Load per-million-token pricing from conf/model_pricing.json.

    Missing entries yield a None cost rather than a guessed one, so an
    unpriced model reads as "unknown" in reports instead of "free".
    """
    global _PRICING_CACHE
    if _PRICING_CACHE is None:
        path = Path(__file__).resolve().parent.parent / "conf" / "model_pricing.json"
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            _PRICING_CACHE = {k.lower(): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception as exc:  # pragma: no cover - config is optional
            logger.debug(f"No model pricing loaded: {exc}")
            _PRICING_CACHE = {}
    return _PRICING_CACHE


# Providers that run on local hardware have no per-token cost.
_FREE_PROVIDERS = {"custom", "local", "ollama"}

# Suffixes that denote a pinned snapshot of the same priced model, e.g.
# "-2026-01-15", "-20260115", "-latest". Anything else is a different model.
_SNAPSHOT_SUFFIX = re.compile(r"|-\d{4}-\d{2}-\d{2}|-\d{6,8}|-latest")


def estimate_cost(model: str, provider: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the USD cost of a completion, or None when pricing is unknown."""
    if (provider or "").lower() in _FREE_PROVIDERS:
        return 0.0
    table = _pricing()
    entry = table.get((model or "").lower())
    if entry is None and model:
        # Fall back to a prefix match only for dated/pinned snapshots, so
        # "gpt-5.1-codex-2026-01-15" picks up "gpt-5.1-codex".
        #
        # The suffix must look like a date or "latest" — never an arbitrary
        # one. Model IDs in a family share prefixes but NOT price tiers:
        # a loose prefix match bills "gpt-5.1-codex" at the "gpt-5" rate and
        # "gpt-5.2-pro" at the "gpt-5.2" rate. Reporting unknown is correct;
        # reporting a confident wrong number is not.
        lowered = model.lower()
        best_key = None
        for key in table:
            if not lowered.startswith(key):
                continue
            if not _SNAPSHOT_SUFFIX.fullmatch(lowered[len(key) :]):
                continue
            if best_key is None or len(key) > len(best_key):
                best_key = key
        if best_key is not None:
            entry = table[best_key]
    if entry is None:
        return None
    per_m_in = entry.get("input_per_1m")
    per_m_out = entry.get("output_per_1m")
    if per_m_in is None and per_m_out is None:
        return None
    return (input_tokens / 1_000_000) * (per_m_in or 0.0) + (output_tokens / 1_000_000) * (per_m_out or 0.0)


# ---------------------------------------------------------------------------
# Per-call context
# ---------------------------------------------------------------------------


@dataclass
class _CallContext:
    tool: str
    client: str | None = None
    continuation: bool = False
    started: float = field(default_factory=time.time)
    models: list[dict[str, Any]] = field(default_factory=list)


_current_call: ContextVar[_CallContext | None] = ContextVar("pal_metrics_call", default=None)


def record_model_usage(model_name: str, provider: str, usage: dict[str, int]) -> None:
    """Attach one model completion to the in-flight tool call.

    Called from ModelResponse construction. No active call context (unit
    tests, direct provider use) means nothing is recorded.
    """
    if not metrics_enabled():
        return
    ctx = _current_call.get()
    if ctx is None:
        return
    try:
        usage = usage or {}
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (inp + out))
        ctx.models.append(
            {
                "ts": time.time(),
                "model": model_name or "unknown",
                "provider": provider or "unknown",
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": total,
                "cost_usd": estimate_cost(model_name, provider, inp, out),
            }
        )
    except Exception as exc:  # pragma: no cover - never break a completion
        logger.debug(f"record_model_usage failed: {exc}")


class track_tool_call:
    """Context manager recording one tool call and its model completions.

    Usage:
        with track_tool_call("chat", client="Claude"):
            ...

    Exceptions propagate untouched; the row is written with status="error".
    """

    def __init__(self, tool: str, client: str | None = None, continuation: bool = False):
        self.tool = tool
        self.client = client
        self.continuation = continuation
        self._token = None
        self._ctx: _CallContext | None = None

    def __enter__(self) -> track_tool_call:
        if metrics_enabled():
            self._ctx = _CallContext(tool=self.tool, client=self.client, continuation=self.continuation)
            self._token = _current_call.set(self._ctx)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _current_call.reset(self._token)
        if self._ctx is not None:
            status = "ok" if exc_type is None else "error"
            err_type = exc_type.__name__ if exc_type else None
            err_msg = str(exc)[:500] if exc else None
            self._flush(self._ctx, status, err_type, err_msg)
        return False  # never suppress

    @staticmethod
    def _flush(ctx: _CallContext, status: str, err_type: str | None, err_msg: str | None) -> None:
        try:
            duration_ms = int((time.time() - ctx.started) * 1000)
            conn = connect()
            try:
                cur = conn.execute(
                    "INSERT INTO calls (ts, tool, client, continuation, duration_ms, status, error_type, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ctx.started, ctx.tool, ctx.client, int(ctx.continuation), duration_ms, status, err_type, err_msg),
                )
                call_id = cur.lastrowid
                if ctx.models:
                    conn.executemany(
                        "INSERT INTO model_calls (call_id, ts, tool, model, provider, input_tokens, "
                        "output_tokens, total_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                call_id,
                                m["ts"],
                                ctx.tool,
                                m["model"],
                                m["provider"],
                                m["input_tokens"],
                                m["output_tokens"],
                                m["total_tokens"],
                                m["cost_usd"],
                            )
                            for m in ctx.models
                        ],
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - metrics must not break calls
            logger.debug(f"Failed to persist call metrics: {exc}")
