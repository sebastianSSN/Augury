"""Tests for /auth/* endpoints."""
import pytest


def test_register_success(client):
    r = client.post("/auth/register", json={"email": "new@example.com", "password": "secure123"})
    assert r.status_code == 201
    data = r.json()
    assert data["email"] == "new@example.com"
    assert data["is_active"] is True
    assert "id" in data
    assert "hashed_password" not in data


def test_register_duplicate_email(client):
    payload = {"email": "dup@example.com", "password": "secure123"}
    client.post("/auth/register", json=payload)
    r = client.post("/auth/register", json=payload)
    assert r.status_code == 400
    assert "registrado" in r.json()["detail"]


def test_register_short_password(client):
    r = client.post("/auth/register", json={"email": "short@example.com", "password": "abc"})
    assert r.status_code == 400


def test_login_success(client):
    client.post("/auth/register", json={"email": "login@example.com", "password": "password123"})
    r = client.post("/auth/login", json={"email": "login@example.com", "password": "password123"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    client.post("/auth/register", json={"email": "wp@example.com", "password": "correct123"})
    r = client.post("/auth/login", json={"email": "wp@example.com", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_email(client):
    r = client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever"})
    assert r.status_code == 401


def test_me_authenticated(client):
    client.post("/auth/register", json={"email": "me@example.com", "password": "password123"})
    r = client.post("/auth/login", json={"email": "me@example.com", "password": "password123"})
    token = r.json()["access_token"]

    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "me@example.com"


def test_me_no_token(client):
    r = client.get("/auth/me")
    assert r.status_code in (401, 403)


def test_me_invalid_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-valid-token"})
    assert r.status_code == 401
