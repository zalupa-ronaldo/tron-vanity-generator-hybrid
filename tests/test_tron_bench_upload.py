import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


MODULE_PATH = Path(__file__).parents[1] / "deploy" / "tron_bench_upload.py"


@pytest.fixture()
def upload_service(tmp_path):
    spec = importlib.util.spec_from_file_location("tron_bench_upload_test_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.ROOT = tmp_path / "reports"
    module.TOKEN_FILE = tmp_path / "legacy-token"
    module.ADMIN_TOKEN_FILE = tmp_path / "admin-token"
    module.TOKEN_STATE_FILE = tmp_path / "reports" / ".token-state.json"
    module.TOKEN_FILE.write_text("legacy-upload-secret\n", encoding="utf-8")
    module.ADMIN_TOKEN_FILE.write_text("admin-secret\n", encoding="utf-8")

    with TestClient(module.app) as client:
        yield module, client


def test_issue_replaces_and_revoke_disables_upload_token(upload_service):
    module, client = upload_service
    admin = {"X-Tron-Bench-Admin-Token": "admin-secret"}

    assert client.post("/tron-bench-admin/token").status_code == 401

    legacy = client.put(
        "/tron-bench-upload/legacy-run/summary.txt",
        headers={"X-Tron-Bench-Token": "legacy-upload-secret"},
        content=b"legacy\n",
    )
    assert legacy.status_code == 201

    first = client.post("/tron-bench-admin/token", headers=admin)
    assert first.status_code == 201
    first_body = first.json()
    first_id = first_body["token_id"]
    first_token = first_body["token"]
    assert first_body["replaced_token_id"] is None
    assert first.headers["cache-control"] == "no-store"
    assert module.TOKEN_STATE_FILE.exists()

    old_legacy = client.put(
        "/tron-bench-upload/after-issue/summary.txt",
        headers={"X-Tron-Bench-Token": "legacy-upload-secret"},
        content=b"must fail\n",
    )
    assert old_legacy.status_code == 401

    current = client.put(
        "/tron-bench-upload/after-issue/summary.txt",
        headers={"X-Tron-Bench-Token": first_token},
        content=b"works\n",
    )
    assert current.status_code == 201

    second = client.post("/tron-bench-admin/token", headers=admin)
    assert second.status_code == 201
    second_body = second.json()
    second_id = second_body["token_id"]
    second_token = second_body["token"]
    assert second_id != first_id
    assert second_body["replaced_token_id"] == first_id

    replaced = client.put(
        "/tron-bench-upload/replaced/summary.txt",
        headers={"X-Tron-Bench-Token": first_token},
        content=b"must fail\n",
    )
    assert replaced.status_code == 401

    revoked = client.delete(f"/tron-bench-admin/token/{second_id}", headers=admin)
    assert revoked.status_code == 200
    assert revoked.json() == {"ok": True, "token_id": second_id, "revoked": True}

    after_revoke = client.put(
        "/tron-bench-upload/after-revoke/summary.txt",
        headers={"X-Tron-Bench-Token": second_token},
        content=b"must fail\n",
    )
    assert after_revoke.status_code == 401
    assert client.delete(f"/tron-bench-admin/token/{second_id}", headers=admin).status_code == 404


def test_legacy_token_still_works_until_admin_state_exists(upload_service):
    _, client = upload_service
    response = client.put(
        "/tron-bench-upload/legacy-only/benchmark.csv",
        headers={"X-Tron-Bench-Token": "legacy-upload-secret"},
        content=b"benchmark\n",
    )
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
