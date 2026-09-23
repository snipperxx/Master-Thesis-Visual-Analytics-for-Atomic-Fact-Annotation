"""Regression guard for BUG-2026-07-06-A.

Every Neo4j driver created in ``src/neo4j_constraints.py`` must go through the
``_driver()`` factory so it carries bounded timeouts and can never hang the UI
on the neo4j-driver defaults. These tests need no running database.
"""
import importlib.util
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "neo4j_constraints.py"


def _load():
    spec = importlib.util.spec_from_file_location("nc_under_test", SRC)
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    return nc


def test_driver_timeout_keys_present():
    nc = _load()
    assert callable(nc._driver)
    assert set(nc._DRIVER_TIMEOUTS) >= {
        "connection_timeout",
        "connection_acquisition_timeout",
        "max_transaction_retry_time",
    }


def test_a_bounded_driver_creation_exists():
    src = SRC.read_text(encoding="utf-8")
    calls = re.findall(r"GraphDatabase\.driver\([^\n]*\)", src)
    assert calls, "expected at least one driver creation"
    assert any("_DRIVER_TIMEOUTS" in c for c in calls), \
        "no bounded driver creation found -- _DRIVER_TIMEOUTS got dropped"


def test_no_unbounded_driver_outside_factory():
    # The only raw, unbounded ``GraphDatabase.driver(uri, auth=(user, pw))`` call
    # allowed is the defensive fallback inside _driver(). status() and
    # run_constraint() must NOT rebuild their own raw driver.
    src = SRC.read_text(encoding="utf-8")
    unbounded = re.findall(r"GraphDatabase\.driver\(uri, auth=\(user, pw\)\)", src)
    assert len(unbounded) <= 1, (
        f"found {len(unbounded)} unbounded GraphDatabase.driver() calls; all "
        "driver creation must go through _driver() (BUG-2026-07-06-A)")


def test_status_tolerates_missing_driver():
    # With the neo4j package absent, status() must return fast WITHOUT touching
    # _driver() (which would raise ImportError).
    nc = _load()
    if not nc.driver_available():
        st = nc.status()
        assert st["connected"] is False
        assert "not installed" in st["message"]
