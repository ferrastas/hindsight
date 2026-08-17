"""Tests for semantic arm sizing: the SQL LIMIT and the ANN candidate list must agree.

The semantic arm over-fetches to compensate for ANN approximation and for the filters
Postgres applies after the index scan. Two things bound how wide it usefully runs: what
fusion will keep (the reranker cap) and what one index scan can return (pgvector's
``hnsw.ef_search``, whose accepted maximum is 1000 and which *is* the result-set size —
a LIMIT above it is unreachable through the index).

Covers:
- :func:`plan_semantic_fetch` across the budget ladder, both bounds, and legacy configs.
- That the resolved plan reaches the SQL (``build_semantic_arm``'s fetch limit) and the
  backend (``SET LOCAL hnsw.ef_search``) as the *same* number.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hindsight_api._vector_index import ann_candidate_list_max, ann_candidate_list_settings
from hindsight_api.config import MIN_SEMANTIC_FETCH
from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.engine.search import retrieval as retrieval_mod
from hindsight_api.engine.search.retrieval import plan_semantic_fetch

# Budget levels as _resolve_thinking_budget maps them (fixed function, the default).
BUDGET_LOW = 100
BUDGET_MID = 300
BUDGET_HIGH = 1000

# The reranker cap fusion truncates to before scoring (DEFAULT_RERANKER_MAX_CANDIDATES).
RERANK_CAP = 300


@pytest.fixture
def config(monkeypatch):
    """A config carrying only the fields the planner reads."""
    cfg = SimpleNamespace(
        semantic_overfetch_factor=2.0,
        reranker_max_candidates=RERANK_CAP,
        # Read by the query builder when the arms are actually assembled.
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# plan_semantic_fetch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "expected_keep", "expected_fetch"),
    [
        (BUDGET_LOW, 100, 200),
        (BUDGET_MID, 300, 600),
        # Above the rerank cap the extra budget buys nothing: fusion would drop the rows.
        (BUDGET_HIGH, RERANK_CAP, 600),
    ],
)
def test_budget_ladder_sizes_both_numbers(config, budget, expected_keep, expected_fetch):
    plan = plan_semantic_fetch(budget, vector_extension="pgvector")
    assert plan.keep_limit == expected_keep
    assert plan.fetch_limit == expected_fetch


def test_fetch_stays_within_what_one_ann_scan_can_return(config):
    """pgvector caps hnsw.ef_search at 1000, so the LIMIT never exceeds it."""
    config.reranker_max_candidates = 2000  # operator raised the rerank cap
    plan = plan_semantic_fetch(2000, vector_extension="pgvector")
    assert plan.fetch_limit == ann_candidate_list_max("pgvector") == 1000
    # Nothing is kept that the index was never asked for.
    assert plan.keep_limit == 1000


def test_backend_without_a_candidate_list_is_not_clamped(config):
    """vchord exposes no per-query knob, so only the budget bounds the arm."""
    config.reranker_max_candidates = 2000
    plan = plan_semantic_fetch(2000, vector_extension="vchord")
    assert ann_candidate_list_max("vchord") is None
    assert plan.fetch_limit == 4000
    assert plan.keep_limit == 2000


def test_explicit_ceiling_overrides_the_configured_cap(config):
    """The caller's budget-resolved cap wins over the flat config value."""
    plan = plan_semantic_fetch(BUDGET_HIGH, candidate_ceiling=1000, vector_extension="pgvector")
    assert plan.keep_limit == 1000
    assert plan.fetch_limit == 1000  # 2000 requested, clamped to the pgvector maximum


def test_ceiling_of_zero_disables_the_bound(config):
    """0 is the 'no cap' convention shared with recall_max_candidates_per_source."""
    plan = plan_semantic_fetch(BUDGET_MID, candidate_ceiling=0, vector_extension="vchord")
    assert plan.keep_limit == BUDGET_MID


def test_small_budgets_keep_a_floor_of_index_coverage(config):
    """A tiny budget still scans enough to survive post-scan filtering."""
    plan = plan_semantic_fetch(20, vector_extension="pgvector")
    assert plan.keep_limit == 20
    assert plan.fetch_limit == MIN_SEMANTIC_FETCH


def test_factor_of_one_disables_the_overfetch(config):
    config.semantic_overfetch_factor = 1.0
    plan = plan_semantic_fetch(BUDGET_MID, vector_extension="pgvector")
    assert plan.fetch_limit == plan.keep_limit == BUDGET_MID


def test_legacy_config_without_the_new_fields_still_plans(monkeypatch):
    """An older embedded profile predating these fields falls back to the defaults."""
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: SimpleNamespace())
    plan = plan_semantic_fetch(BUDGET_MID, vector_extension="pgvector")
    assert plan.keep_limit == 300
    assert plan.fetch_limit == 600


# ---------------------------------------------------------------------------
# The plan reaching SQL and the backend as one number
# ---------------------------------------------------------------------------


class FakeDialect:
    """Captures what the semantic arm asks the index for."""

    def __init__(self):
        self.fetch_limits: list[int] = []

    def build_semantic_arm(self, *, fetch_limit, **kwargs):
        self.fetch_limits.append(fetch_limit)
        return "SELECT 'semantic' AS source"

    def build_bm25_arm(self, **kwargs):
        return "SELECT 'bm25' AS source"

    def prepare_bm25_text(self, tokens, query_text, **kwargs):
        return " | ".join(tokens)


class FakeConn:
    """Records the session settings applied around the query."""

    backend_type = "postgresql"

    def __init__(self):
        self.executed: list[str] = []
        self.in_transaction = False

    def transaction(self):
        conn = self

        class _Txn:
            async def __aenter__(self):
                conn.in_transaction = True

            async def __aexit__(self, *exc):
                conn.in_transaction = False
                return False

        return _Txn()

    async def execute(self, sql, *params):
        self.executed.append(sql)

    async def fetch(self, query, *params):
        # The arms must run inside the transaction that scopes the candidate list,
        # otherwise the SET LOCAL has already reverted.
        assert self.in_transaction, "semantic arms ran outside the candidate-list transaction"
        return []


@pytest.fixture
def fake_query_path(monkeypatch, config):
    dialect = FakeDialect()
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: dialect)
    monkeypatch.setattr(retrieval_mod, "configured_vector_extension", lambda: "pgvector")
    # postgres.search imports these at call time, so the source module is the patch point.
    monkeypatch.setattr("hindsight_api._vector_index.configured_vector_extension", lambda: "pgvector")
    monkeypatch.setattr(retrieval_mod, "fq_table", lambda name: name)
    monkeypatch.setattr(retrieval_mod, "get_current_schema", lambda: None)
    return dialect


async def test_search_sizes_the_candidate_list_to_the_sql_limit(fake_query_path):
    """The SET LOCAL and the arm's LIMIT are the same number — the bug this guards."""
    conn = FakeConn()
    await PostgresMemories({}).search(
        conn=conn,
        bank_id="bank-1",
        fact_types=["world", "experience"],
        query_embedding="[0.0]",
        query_text="alpha beta",
        limit=BUDGET_MID,
    )

    expected = plan_semantic_fetch(BUDGET_MID, vector_extension="pgvector").fetch_limit
    assert conn.executed == [f"SET LOCAL hnsw.ef_search = {expected}"]
    # One arm per fact_type, each asking the index for exactly what the list holds.
    assert fake_query_path.fetch_limits == [expected, expected]


async def test_search_skips_the_setting_for_backends_without_the_knob(monkeypatch, fake_query_path):
    monkeypatch.setattr("hindsight_api._vector_index.configured_vector_extension", lambda: "vchord")
    conn = FakeConn()
    # No transaction is opened, so fetch() must not assert on one.
    conn.in_transaction = True

    await PostgresMemories({}).search(
        conn=conn,
        bank_id="bank-1",
        fact_types=["world"],
        query_embedding="[0.0]",
        query_text="alpha beta",
        limit=BUDGET_MID,
    )

    assert conn.executed == []
    assert ann_candidate_list_settings("vchord", candidates=600) == ()


async def test_search_honours_the_callers_candidate_ceiling(fake_query_path):
    """A HIGH budget with the default rerank cap narrows to what fusion keeps."""
    conn = FakeConn()
    await PostgresMemories({}).search(
        conn=conn,
        bank_id="bank-1",
        fact_types=["world"],
        query_embedding="[0.0]",
        query_text="alpha beta",
        limit=BUDGET_HIGH,
        candidate_ceiling=RERANK_CAP,
    )

    assert conn.executed == ["SET LOCAL hnsw.ef_search = 600"]
    assert fake_query_path.fetch_limits == [600]


def test_candidate_list_request_is_clamped_to_the_backend_maximum():
    """Even an out-of-range request produces a value the GUC accepts."""
    assert ann_candidate_list_settings("pgvector", candidates=99_999) == (("hnsw.ef_search", "1000"),)
    assert ann_candidate_list_settings("pgvector", candidates=0) == (("hnsw.ef_search", "1"),)
