"""Tests for utilisation metrics recording and cost estimation."""

import sqlite3

import pytest

from providers.shared.model_response import ModelResponse
from providers.shared.provider_type import ProviderType
from utils import metrics


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point metrics at a throwaway database."""
    monkeypatch.setenv("PAL_METRICS_DB", str(tmp_path / "m.db"))
    monkeypatch.setattr(metrics, "_initialised", False)
    yield tmp_path / "m.db"


def rows(db_path, table):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


class TestRecording:
    def test_successful_call_records_tokens(self, db):
        with metrics.track_tool_call("chat", client="Claude"):
            ModelResponse(
                content="hi",
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                model_name="qwen2.5-coder:32b",
                provider=ProviderType.CUSTOM,
            )
        (call,) = rows(db, "calls")
        (model_call,) = rows(db, "model_calls")
        assert call["tool"] == "chat"
        assert call["status"] == "ok"
        assert model_call["total_tokens"] == 150
        assert model_call["call_id"] == call["id"]

    def test_multiple_models_in_one_call(self, db):
        """A consensus-style call attributes each completion separately."""
        with metrics.track_tool_call("consensus"):
            for name in ("model-a", "model-b"):
                ModelResponse(content="x", usage={"total_tokens": 10}, model_name=name)
        assert len(rows(db, "calls")) == 1
        assert {r["model"] for r in rows(db, "model_calls")} == {"model-a", "model-b"}

    def test_failure_is_recorded_and_exception_propagates(self, db):
        with pytest.raises(ValueError):
            with metrics.track_tool_call("debug"):
                raise ValueError("boom")
        (call,) = rows(db, "calls")
        assert call["status"] == "error"
        assert call["error_type"] == "ValueError"

    def test_no_context_is_a_noop(self, db):
        """Provider use outside a tracked call must not write rows."""
        ModelResponse(content="x", usage={"total_tokens": 999}, model_name="orphan")
        conn = sqlite3.connect(str(db))
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        finally:
            conn.close()
        assert not tables or not rows(db, "model_calls")

    def test_disabled_records_nothing(self, db, monkeypatch):
        monkeypatch.setenv("PAL_METRICS_ENABLED", "false")
        with metrics.track_tool_call("chat"):
            ModelResponse(content="x", usage={"total_tokens": 5}, model_name="m")
        assert not db.exists()


class TestCostEstimation:
    @pytest.fixture(autouse=True)
    def pricing(self, monkeypatch):
        monkeypatch.setattr(
            metrics,
            "_PRICING_CACHE",
            {
                "gpt-5": {"input_per_1m": 1.25, "output_per_1m": 10.00},
                "gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.00},
                "gpt-5.2": {"input_per_1m": 1.75, "output_per_1m": 14.00},
            },
        )

    def test_exact_match(self):
        assert metrics.estimate_cost("gpt-5-mini", "openai", 1_000_000, 1_000_000) == pytest.approx(2.25)

    def test_local_provider_is_free(self):
        assert metrics.estimate_cost("anything", "custom", 999_999, 999_999) == 0.0

    def test_unknown_model_is_none_not_zero(self):
        """Unknown pricing must be distinguishable from genuinely free."""
        assert metrics.estimate_cost("mystery-model", "openai", 1000, 1000) is None

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5-mini",  # must NOT fall back to the shorter "gpt-5" key
            "gpt-5.2",  # must NOT fall back to "gpt-5"
        ],
    )
    def test_longest_key_wins(self, model):
        exact = metrics._pricing()[model]["input_per_1m"]
        assert metrics.estimate_cost(model, "openai", 1_000_000, 0) == pytest.approx(exact)

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5-codex",  # a different model, not a gpt-5 snapshot
            "gpt-5.1-codex",
            "gpt-5.2-pro",  # a pro tier, priced far above gpt-5.2
            "gpt-5-turbo",
        ],
    )
    def test_sibling_models_do_not_borrow_prices(self, model):
        """Sharing a prefix is not sharing a price tier."""
        assert metrics.estimate_cost(model, "openai", 1_000_000, 0) is None

    @pytest.mark.parametrize(
        "model",
        ["gpt-5-mini-2026-01-15", "gpt-5-mini-20260115", "gpt-5-mini-latest"],
    )
    def test_dated_snapshots_resolve(self, model):
        assert metrics.estimate_cost(model, "openai", 1_000_000, 0) == pytest.approx(0.25)
