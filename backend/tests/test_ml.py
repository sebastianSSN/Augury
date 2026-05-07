"""Tests for ML endpoints."""
import pytest
from tests.conftest import make_csv, train_and_wait


# ── /analyze ──────────────────────────────────────────────────────────────────

def test_analyze_success(auth_client):
    client, headers = auth_client
    r = client.post("/analyze", files={"file": ("data.csv", make_csv(), "text/csv")}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["rows"] == 100
    assert data["columns"] == 3
    assert len(data["column_info"]) == 3


def test_analyze_unauthenticated(client):
    r = client.post("/analyze", files={"file": ("data.csv", make_csv(), "text/csv")})
    assert r.status_code in (401, 403)


def test_analyze_too_few_rows(auth_client):
    client, headers = auth_client
    r = client.post("/analyze", files={"file": ("tiny.csv", b"a,b\n1,2\n3,4", "text/csv")}, headers=headers)
    assert r.status_code == 400


def test_analyze_invalid_csv(auth_client):
    client, headers = auth_client
    r = client.post("/analyze", files={"file": ("bad.csv", b"\xff\xfe garbage", "text/csv")}, headers=headers)
    assert r.status_code == 400


# ── /suggest-drops ────────────────────────────────────────────────────────────

def test_suggest_drops(auth_client):
    client, headers = auth_client
    r = client.post("/suggest-drops", files={"file": ("data.csv", make_csv(), "text/csv")}, headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json()["suggestions"], list)


# ── /train (async) ────────────────────────────────────────────────────────────

def test_train_queues_job(auth_client):
    client, headers = auth_client
    r = client.post(
        "/train",
        files={"file": ("data.csv", make_csv(), "text/csv")},
        data={"target_col": "label"},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_train_status_done(auth_client):
    client, headers = auth_client
    result = train_and_wait(client, headers)
    assert result["status"] == "done"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "metrics" in result
    assert "feature_uniques" in result


def test_train_missing_target(auth_client):
    client, headers = auth_client
    r = client.post(
        "/train",
        files={"file": ("data.csv", make_csv(), "text/csv")},
        data={"target_col": "nonexistent_col"},
        headers=headers,
    )
    assert r.status_code == 400


def test_train_no_features_left(auth_client):
    client, headers = auth_client
    r = client.post(
        "/train",
        files={"file": ("data.csv", make_csv(), "text/csv")},
        data={"target_col": "label", "drop_cols": "age,income"},
        headers=headers,
    )
    assert r.status_code == 400


def test_train_unauthenticated(client):
    r = client.post(
        "/train",
        files={"file": ("data.csv", make_csv(), "text/csv")},
        data={"target_col": "label"},
    )
    assert r.status_code in (401, 403)


# ── /model-info ───────────────────────────────────────────────────────────────

def test_model_info_after_training(auth_client):
    client, headers = auth_client
    train_and_wait(client, headers)
    r = client.get("/model-info", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["trained"] is True
    assert "accuracy" in data
    assert "feature_importances" in data


# ── /predict-single ───────────────────────────────────────────────────────────

def test_predict_single_success(auth_client):
    client, headers = auth_client
    train_and_wait(client, headers)
    r = client.post("/predict-single", json={"age": 30, "income": 50000}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("yes", "no")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_single_missing_feature(auth_client):
    client, headers = auth_client
    train_and_wait(client, headers)
    r = client.post("/predict-single", json={"age": 30}, headers=headers)
    assert r.status_code == 400


def test_predict_single_no_model(client):
    client.post("/auth/register", json={"email": "nomodel@example.com", "password": "pass1234"})
    r = client.post("/auth/login", json={"email": "nomodel@example.com", "password": "pass1234"})
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = client.post("/predict-single", json={"x": 1}, headers=headers)
    assert r.status_code == 400


# ── /model DELETE ─────────────────────────────────────────────────────────────

def test_delete_model(auth_client):
    client, headers = auth_client
    train_and_wait(client, headers)
    r = client.delete("/model", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"
    assert client.get("/model-info", headers=headers).json()["trained"] is False


# ── /algorithms ───────────────────────────────────────────────────────────────

def test_list_algorithms(client):
    r = client.get("/algorithms")
    assert r.status_code == 200
    ids = [a["id"] for a in r.json()]
    assert "random_forest" in ids
    assert "gradient_boosting" in ids
    assert "logistic_regression" in ids


# ── /train with algorithm param ───────────────────────────────────────────────

def test_train_gradient_boosting(auth_client):
    client, headers = auth_client
    result = train_and_wait(client, headers, algorithm="gradient_boosting")
    assert result["status"] == "done"
    assert result["metrics"]["algorithm"] == "gradient_boosting"


def test_train_logistic_regression(auth_client):
    client, headers = auth_client
    result = train_and_wait(client, headers, algorithm="logistic_regression")
    assert result["status"] == "done"
    assert result["metrics"]["algorithm"] == "logistic_regression"


def test_train_invalid_algorithm(auth_client):
    client, headers = auth_client
    r = client.post(
        "/train",
        files={"file": ("data.csv", make_csv(), "text/csv")},
        data={"target_col": "label", "algorithm": "made_up_algo"},
        headers=headers,
    )
    assert r.status_code == 400


# ── /predict-csv ──────────────────────────────────────────────────────────────

def test_predict_csv_success(auth_client):
    client, headers = auth_client
    train_and_wait(client, headers)
    r = client.post("/predict-csv", files={"file": ("data.csv", make_csv(), "text/csv")}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    lines = r.content.decode().strip().split("\n")
    assert len(lines) == 101  # header + 100 rows
    assert "prediction" in lines[0]
    assert "confidence" in lines[0]


def test_predict_csv_no_model(client):
    client.post("/auth/register", json={"email": "nomodel_csv@example.com", "password": "pass1234"})
    r = client.post("/auth/login", json={"email": "nomodel_csv@example.com", "password": "pass1234"})
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    r = client.post("/predict-csv", files={"file": ("data.csv", make_csv(), "text/csv")}, headers=headers)
    assert r.status_code == 400


# ── User isolation ────────────────────────────────────────────────────────────

def test_users_are_isolated(client):
    def register_train(email: str) -> dict:
        client.post("/auth/register", json={"email": email, "password": "password123"})
        r = client.post("/auth/login", json={"email": email, "password": "password123"})
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        train_and_wait(client, headers)
        return headers

    headers_a = register_train("iso_userA@example.com")
    headers_b = register_train("iso_userB@example.com")

    client.delete("/model", headers=headers_a)
    assert client.get("/model-info", headers=headers_a).json()["trained"] is False
    assert client.get("/model-info", headers=headers_b).json()["trained"] is True
