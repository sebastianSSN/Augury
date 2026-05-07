"""
Shared pytest fixtures.

- SQLite in-memory  → no PostgreSQL needed
- Celery eager mode → no Redis needed, tasks run synchronously
"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Environment must be set BEFORE any app import ─────────────────────────────
_tmp_model_dir = tempfile.mkdtemp()
os.environ.setdefault("MODEL_DIR",    _tmp_model_dir)
os.environ.setdefault("SECRET_KEY",   "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("REDIS_URL",    "memory://")
os.environ.setdefault("LOG_FORMAT",   "text")

# ── Celery eager mode (tasks run synchronously, no Redis needed) ───────────────
from celery_app import celery_app
celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,   # Store results even in eager mode
    result_backend="cache+memory://",
    broker_url="memory://",
)

# ── Disable rate limiting in tests ───────────────────────────────────────────
# All tests share "testclient" as source IP; patch the check method class-wide.
# Must set request.state.view_rate_limit so SlowAPIMiddleware doesn't crash.
from slowapi import Limiter as _Limiter

def _noop_check(self, request, endpoint_func, in_middleware):
    if in_middleware:
        request.state.view_rate_limit = None

_Limiter._check_request_limit = _noop_check

# ── SQLite test database ───────────────────────────────────────────────────────
from database import Base, get_db
from main import app

_engine      = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = _override_get_db
    # Tasks also need the SQLite DB — patch SessionLocal used inside tasks.py
    import database
    database.SessionLocal = _TestSession
    yield
    Base.metadata.drop_all(bind=_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_csv(rows: int = 100) -> bytes:
    """Simple binary classification CSV for tests."""
    lines = ["age,income,label"]
    for i in range(rows):
        age    = 20 + (i % 50)
        income = 20000 + (i * 500)
        label  = "yes" if i % 3 != 0 else "no"
        lines.append(f"{age},{income},{label}")
    return "\n".join(lines).encode()


@pytest.fixture
def csv_bytes():
    return make_csv()


@pytest.fixture
def auth_client(client):
    """Returns (TestClient, auth_headers) for a freshly authenticated user."""
    email, pw = "test@example.com", "password123"
    client.post("/auth/register", json={"email": email, "password": pw})
    r = client.post("/auth/login", json={"email": email, "password": pw})
    token = r.json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def train_and_wait(client, headers, csv_data=None, target_col="label", drop_cols="", algorithm="random_forest") -> dict:
    """
    Helper: POST /train → poll /train/status until done.
    Returns the final status response (status='done', metrics, feature_uniques).
    With Celery eager mode the task completes synchronously,
    so the status endpoint returns 'done' on the first poll.
    """
    data = csv_data or make_csv()
    r = client.post(
        "/train",
        files={"file": ("data.csv", data, "text/csv")},
        data={"target_col": target_col, "drop_cols": drop_cols, "algorithm": algorithm},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # Poll until done (eager mode: should be immediate)
    for _ in range(10):
        sr = client.get(f"/train/status/{job_id}", headers=headers)
        assert sr.status_code == 200
        status_data = sr.json()
        if status_data["status"] in ("done", "failed"):
            return status_data
    raise TimeoutError("Training did not complete in time (eager mode)")
