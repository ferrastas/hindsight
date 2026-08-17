"""The semantic arm's ANN search is sized for the query, not fixed at connection init.

An ANN scan explores a bounded candidate list and returns what it found, so that list
— not the SQL LIMIT — decides how many rows the arm can come back with. pgvector's is
``hnsw.ef_search``, pinned at 200 for the connection's lifetime by the pool's init
callback, which silently capped every recall at ~200 dense candidates however large the
budget. ``PostgresMemories.search`` now sizes it to the rows the query asks for.

Covers:
- :func:`ann_candidate_list_settings`: clamping, and staying silent when the
  connection's standing value already covers the request.
- That ``search`` issues the wider value for a large budget, issues nothing for a
  small one, and does not open a transaction to do it.
- That the semantic arms ask the index for exactly ``limit`` rows (no over-fetch).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hindsight_api._vector_index import ann_candidate_list_settings
from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.engine.search import retrieval as retrieval_mod

# Budget levels as _resolve_thinking_budget maps them (fixed function, the default).
BUDGET_LOW = 100
BUDGET_MID = 300
BUDGET_HIGH = 1000

# What the pool's init callback leaves on every connection (_ANN_TUNING_HIGH_RECALL).
STANDING_EF_SEARCH = 200


# ---------------------------------------------------------------------------
# ann_candidate_list_settings
# ---------------------------------------------------------------------------


def test_widens_the_candidate_list_when_the_query_needs_more():
    assert ann_candidate_list_settings("pgvector", candidates=600) == (("hnsw.ef_search", "600"),)


def test_stays_silent_when_the_standing_value_already_covers_it():
    """Small budgets keep today's behaviour and cost no extra round trip."""
    assert ann_candidate_list_settings("pgvector", candidates=STANDING_EF_SEARCH) == ()
    assert ann_candidate_list_settings("pgvector", candidates=50) == ()


def test_request_is_clamped_to_the_backend_maximum():
    """pgvector declares hnsw.ef_search valid over 1..1000; SET rejects more."""
    assert ann_candidate_list_settings("pgvector", candidates=1000) == (("hnsw.ef_search", "1000"),)
    assert ann_candidate_list_settings("pgvector", candidates=99_999) == (("hnsw.ef_search", "1000"),)


def test_backends_without_the_knob_get_no_setting():
    """vchord / diskann / scann expose no per-query candidate list."""
    for ext in ("vchord", "pgvectorscale", "pg_diskann", "scann"):
        assert ann_candidate_list_settings(ext, candidates=5000) == ()


# ---------------------------------------------------------------------------
# What search() actually issues
# ---------------------------------------------------------------------------


class FakeDialect:
    """Captures what each semantic arm asks the index for."""

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
    """Records every statement, and whether a transaction was ever opened."""

    backend_type = "postgresql"

    def __init__(self):
        self.statements: list[tuple[str, tuple]] = []
        self.transactions = 0

    def transaction(self):
        self.transactions += 1
        raise AssertionError("recall must not run inside an explicit transaction")

    async def execute(self, sql, *params):
        self.statements.append((sql, params))
        return None

    async def fetch(self, query, *params):
        self.statements.append((query, params))
        return []

    def settings_applied(self) -> list[tuple]:
        """The GUC writes, as (name, value) pairs — issued via set_config, not SET."""
        return [params for sql, params in self.statements if "set_config" in sql]


@pytest.fixture
def search_path(monkeypatch):
    """Stub the query builder and config so only the sizing behaviour is under test."""
    dialect = FakeDialect()
    config = SimpleNamespace(
        # Read by the query builder when the arms are assembled.
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: dialect)
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "fq_table", lambda name: name)
    monkeypatch.setattr(retrieval_mod, "get_current_schema", lambda: None)
    # search() imports these at call time, so the source modules are the patch points.
    monkeypatch.setattr("hindsight_api.config.get_config", lambda: config)
    monkeypatch.setattr("hindsight_api._vector_index.configured_vector_extension", lambda: "pgvector")
    return dialect


async def _search(conn, limit: int, fact_types: list[str] | None = None):
    await PostgresMemories({}).search(
        conn=conn,
        bank_id="bank-1",
        fact_types=fact_types or ["world", "experience"],
        query_embedding="[0.0]",
        query_text="alpha beta",
        limit=limit,
    )


async def test_large_budget_widens_the_search(search_path):
    """The list is sized to the rows asked for, which the standing 200 cannot cover."""
    conn = FakeConn()
    await _search(conn, BUDGET_MID)

    assert conn.settings_applied() == [("hnsw.ef_search", str(BUDGET_MID))]
    assert conn.transactions == 0


async def test_small_budget_issues_no_setting(search_path):
    """The standing 200 already covers a budget of 100 — and is never narrowed to it."""
    conn = FakeConn()
    await _search(conn, BUDGET_LOW)

    assert conn.settings_applied() == []


async def test_widening_is_capped_at_the_backend_maximum(search_path):
    conn = FakeConn()
    await _search(conn, BUDGET_HIGH)

    assert conn.settings_applied() == [("hnsw.ef_search", "1000")]


async def test_arms_ask_for_exactly_the_rows_they_keep(search_path):
    """No row over-fetch: the arms are ordered by distance, so trimming N of 5N in
    Python returned precisely what LIMIT N would have."""
    conn = FakeConn()
    await _search(conn, BUDGET_MID)

    assert search_path.fetch_limits == [BUDGET_MID, BUDGET_MID]  # one arm per fact_type


async def test_oracle_gets_no_pgvector_setting(search_path):
    conn = FakeConn()
    conn.backend_type = "oracle"
    await _search(conn, BUDGET_MID)

    assert conn.settings_applied() == []
