"""Regression test proving complete test-data isolation for SQLite evaluation storage.

Verifies:
1. Direct construction of StorageManager() with default arguments is redirected to an isolated temporary database.
2. Module-level singleton `engine = EvaluationEngine()` in `app.routers.evaluation` is redirected to an isolated temporary database.
3. Importing application services and running evaluations via FastAPI endpoints NEVER alters the real `bolibench_evaluation.db`.
4. The real database's file size, mtime, journal mode, schema objects, and row counts remain 100% invariant.
5. All operations run against temporary storage; zero writes are performed against the real database.
"""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from httpx import ASGITransport, AsyncClient

from app.evaluation.storage import StorageManager
from app.main import app
from app.routers.evaluation import engine
from tests.test_storage_concurrency_and_migrations import make_dummy_result


def get_real_db_snapshot():
    """Captures an immutable snapshot of the real bolibench_evaluation.db file and schema using read-only access."""
    real_db = Path(__file__).resolve().parent.parent / "bolibench_evaluation.db"
    if not real_db.exists():
        pytest.skip("bolibench_evaluation.db does not exist locally.")

    wal_file = real_db.parent / f"{real_db.name}-wal"
    shm_file = real_db.parent / f"{real_db.name}-shm"

    # Capture filesystem stat
    db_stat = real_db.stat()
    wal_bytes = wal_file.stat().st_size if wal_file.exists() else 0
    snapshot = {
        "path": real_db,
        "size": db_stat.st_size,
        "mtime_ns": db_stat.st_mtime_ns,
        "wal_bytes": wal_bytes,
    }

    # Query schema and row counts using strict SQLite read-only mode (?mode=ro)
    uri = f"file:{real_db.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode")
        snapshot["journal_mode"] = cursor.fetchone()[0].lower()

        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type in ('table', 'index') ORDER BY name")
        snapshot["schema_objects"] = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
        snapshot["evaluation_runs_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM schema_migrations")
        snapshot["schema_migrations_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT * FROM schema_migrations ORDER BY version")
        snapshot["applied_migrations"] = cursor.fetchall()
    finally:
        conn.close()

    return snapshot


def test_direct_storagemanager_construction_uses_isolated_db():
    """Proves that direct StorageManager() construction with default arguments resolves to an isolated temp DB."""
    before = get_real_db_snapshot()
    real_db_path = before["path"].resolve()

    # Direct construction with no arguments
    storage = StorageManager()

    # Must NOT point to the real database file
    assert Path(storage.db_path).resolve() != real_db_path, (
        f"StorageManager() defaulted to real database: {storage.db_path}"
    )
    # Must point to an isolated test/temp database
    assert any(sub in storage.db_path.lower() for sub in ("temp", "tmp", "pytest", "isolated")), (
        f"StorageManager() db_path does not indicate a temporary path: {storage.db_path}"
    )

    # Perform a write to verify it operates cleanly on the temporary database
    test_run = make_dummy_result("run_isolation_direct_001")
    storage.save_run(test_run)
    retrieved = storage.get_run("run_isolation_direct_001")
    assert retrieved is not None
    assert retrieved.run_id == "run_isolation_direct_001"

    # Verify the real database is 100% untouched
    after = get_real_db_snapshot()
    assert after["evaluation_runs_count"] == before["evaluation_runs_count"], "Real database evaluation_runs count changed!"
    assert after["schema_migrations_count"] == before["schema_migrations_count"], "Real database migrations count changed!"
    assert after["schema_objects"] == before["schema_objects"], "Real database schema modified!"
    assert after["journal_mode"] == before["journal_mode"], "Real database journal mode modified!"
    assert after["size"] == before["size"], "Real database file size changed!"
    assert after["mtime_ns"] == before["mtime_ns"], "Real database mtime changed!"
    assert after["wal_bytes"] == before["wal_bytes"], "Real database WAL file received writes!"


@pytest.mark.asyncio
async def test_module_level_engine_singleton_uses_isolated_db(async_client):
    """Proves that the module-level singleton `engine.storage` in `app.routers.evaluation` is redirected to an isolated temp DB."""
    before = get_real_db_snapshot()
    real_db_path = before["path"].resolve()

    # The router's singleton engine must NOT point to the real database
    assert Path(engine.storage.db_path).resolve() != real_db_path, (
        f"Router engine.storage points to real database: {engine.storage.db_path}"
    )
    assert any(sub in engine.storage.db_path.lower() for sub in ("temp", "tmp", "pytest", "isolated"))

    # Execute a mock evaluation through the FastAPI HTTP client
    resp = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    # Verify the record exists in the test engine's storage
    test_run = engine.get_run(run_id)
    assert test_run is not None

    # Verify the real database remains strictly unchanged
    after = get_real_db_snapshot()
    assert after["evaluation_runs_count"] == before["evaluation_runs_count"]
    assert after["schema_migrations_count"] == before["schema_migrations_count"]
    assert after["schema_objects"] == before["schema_objects"]
    assert after["journal_mode"] == before["journal_mode"]
    assert after["size"] == before["size"]
    assert after["mtime_ns"] == before["mtime_ns"]
    assert after["wal_bytes"] == before["wal_bytes"]


def test_importing_application_in_subprocess_does_not_alter_real_db(tmp_path):
    """Proves that a fresh Python process importing app.main under test runner flags redirects to isolated storage and leaves real DB untouched."""
    before = get_real_db_snapshot()
    repo_root = Path(__file__).resolve().parent.parent

    temp_db_path = tmp_path / "subprocess_test_isolated.db"

    # Command simulates a test execution environment importing the application
    cmd = [
        sys.executable,
        "-c",
        (
            "import os, sys; "
            f"os.environ['BOLIBENCH_DB_PATH'] = r'{temp_db_path}'; "
            "from app.main import app; "
            "from app.routers.evaluation import engine; "
            "from app.evaluation.storage import StorageManager; "
            "s = StorageManager(); "
            f"assert s.db_path == r'{temp_db_path}'; "
            f"assert engine.storage.db_path == r'{temp_db_path}'; "
        ),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(cmd, env=env, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"

    # Verify the real database was not modified at all
    after = get_real_db_snapshot()
    assert after["evaluation_runs_count"] == before["evaluation_runs_count"]
    assert after["schema_migrations_count"] == before["schema_migrations_count"]
    assert after["schema_objects"] == before["schema_objects"]
    assert after["journal_mode"] == before["journal_mode"]
    assert after["size"] == before["size"]
    assert after["mtime_ns"] == before["mtime_ns"]
    assert after["wal_bytes"] == before["wal_bytes"]


def test_pytest_fallback_without_env_var_in_subprocess(tmp_path):
    """Proves that running pytest without preconfigured BOLIBENCH_DB_PATH falls back to isolated temp storage.

    Verifies:
    1. A subprocess executing pytest with --noconftest (preventing conftest from setting BOLIBENCH_DB_PATH)
       relies solely on the engine's built-in fallback guard ('pytest' in sys.modules).
    2. Default StorageManager() construction automatically resolves to a temporary database.
    3. Module-level EvaluationEngine singleton storage resolves to a temporary database.
    4. Mock evaluation and write operations succeed against the isolated fallback database.
    5. The real production database remains 100% untouched.
    """
    before = get_real_db_snapshot()
    repo_root = Path(__file__).resolve().parent.parent

    # Create an isolated standalone test file inside tmp_path
    standalone_test_file = tmp_path / "test_fallback_isolated.py"
    standalone_test_file.write_text(
        (
            "import os\n"
            "from pathlib import Path\n"
            "from app.evaluation.storage import StorageManager, PROD_DB_PATH\n"
            "from app.routers.evaluation import engine\n"
            "\n"
            "def test_fallback_uses_temp_storage():\n"
            "    # 1. Ensure BOLIBENCH_DB_PATH is absent\n"
            "    assert 'BOLIBENCH_DB_PATH' not in os.environ\n"
            "\n"
            "    # 2. Verify direct StorageManager construction selects temp storage\n"
            "    storage = StorageManager()\n"
            "    assert Path(storage.db_path).resolve() != PROD_DB_PATH.resolve(), f'Defaulted to prod: {storage.db_path}'\n"
            "    assert any(sub in storage.db_path.lower() for sub in ('temp', 'tmp', 'pytest', 'isolated'))\n"
            "\n"
            "    # 3. Verify module-level engine singleton selects temp storage\n"
            "    assert Path(engine.storage.db_path).resolve() != PROD_DB_PATH.resolve()\n"
            "    assert any(sub in engine.storage.db_path.lower() for sub in ('temp', 'tmp', 'pytest', 'isolated'))\n"
            "\n"
            "    # 4. Perform an evaluation run to prove writes target only the temp database\n"
            "    result = engine.run_mock_evaluation('pair_qty_001', scenario='ideal_preserved')\n"
            "    assert result.run_id is not None\n"
            "    retrieved = engine.storage.get_run(result.run_id)\n"
            "    assert retrieved is not None\n"
            "    assert retrieved.run_id == result.run_id\n"
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("BOLIBENCH_DB_PATH", None)
    env["PYTHONPATH"] = str(repo_root)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        str(standalone_test_file),
    ]

    result = subprocess.run(cmd, env=env, cwd=str(repo_root), capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess pytest run failed:\nStdout: {result.stdout}\nStderr: {result.stderr}"

    # Verify the real database is 100% untouched
    after = get_real_db_snapshot()
    assert after["evaluation_runs_count"] == before["evaluation_runs_count"]
    assert after["schema_migrations_count"] == before["schema_migrations_count"]
    assert after["schema_objects"] == before["schema_objects"]
    assert after["journal_mode"] == before["journal_mode"]
    assert after["size"] == before["size"]
    assert after["mtime_ns"] == before["mtime_ns"]
    assert after["wal_bytes"] == before["wal_bytes"]
