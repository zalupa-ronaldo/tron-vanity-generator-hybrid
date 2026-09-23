#!/usr/bin/env python3
"""Authenticated FastAPI receiver for no-wallet benchmark reports."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse


ROOT = Path(os.environ.get("TRON_BENCH_ROOT", "/srv/tron-benchmark-reports"))
TOKEN_FILE = Path(os.environ.get("TRON_BENCH_TOKEN_FILE", "/etc/tron-bench-upload/token"))
ADMIN_TOKEN_FILE = Path(
    os.environ.get("TRON_BENCH_ADMIN_TOKEN_FILE", "/etc/tron-bench-upload/admin-token")
)
TOKEN_STATE_FILE = Path(
    os.environ.get("TRON_BENCH_TOKEN_STATE_FILE", "/srv/tron-benchmark-reports/.token-state.json")
)
MAX_BYTES = 2 * 1024 * 1024
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$")
ALLOWED_FILES = {"summary.txt", "benchmark.csv"}
AUTH_LOCK = threading.RLock()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"ok": False, "error": message},
        headers={"Cache-Control": "no-store"},
    )


def json_response(status: int, content: dict[str, object]) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, headers={"Cache-Control": "no-store"})


def _read_secret(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    if len(value) > 512:
        return ""
    return value.strip()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
            os.fchmod(out.fileno(), 0o600)
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _load_token_state() -> dict[str, object] | None:
    try:
        raw = TOKEN_STATE_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, UnicodeError):
        raise ValueError("token state is unreadable")
    if len(raw) > 4096:
        raise ValueError("token state is oversized")
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("token state is invalid") from exc
    if not isinstance(state, dict):
        raise ValueError("token state is invalid")
    token_id = state.get("token_id")
    token = state.get("token")
    enabled = state.get("enabled")
    if not isinstance(token_id, str) or not TOKEN_ID_RE.fullmatch(token_id):
        raise ValueError("token state is invalid")
    if not isinstance(token, str) or len(token) > 512:
        raise ValueError("token state is invalid")
    if not isinstance(enabled, bool):
        raise ValueError("token state is invalid")
    if enabled and not token:
        raise ValueError("token state is invalid")
    return state


def _current_upload_token() -> str:
    try:
        state = _load_token_state()
    except ValueError:
        return ""
    if state is not None:
        if not state["enabled"]:
            return ""
        return str(state["token"])
    return _read_secret(TOKEN_FILE)


def _admin_authorized(header: str) -> bool:
    expected = _read_secret(ADMIN_TOKEN_FILE)
    return bool(expected) and bool(header) and hmac.compare_digest(header, expected)


@app.get("/tron-bench-upload/health")
async def health() -> dict[str, object]:
    return {"ok": True, "service": "tron-bench-upload"}


@app.post("/tron-bench-admin/token")
async def issue_token(
    x_tron_bench_admin_token: str = Header(default=""),
) -> JSONResponse:
    if not _admin_authorized(x_tron_bench_admin_token):
        return json_error(401, "unauthorized")

    with AUTH_LOCK:
        try:
            previous = _load_token_state()
        except ValueError:
            return json_error(503, "token state is invalid")
        token_id = secrets.token_hex(16)
        token = secrets.token_urlsafe(32)
        state = {
            "version": 1,
            "enabled": True,
            "token_id": token_id,
            "token": token,
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        }
        try:
            _atomic_write(TOKEN_STATE_FILE, json.dumps(state, separators=(",", ":")) + "\n")
        except OSError:
            return json_error(503, "token state could not be written")

    return json_response(
        201,
        {
            "ok": True,
            "token_id": token_id,
            "token": token,
            "replaced_token_id": previous.get("token_id") if previous else None,
            "warning": "token is returned once; store it securely",
        },
    )


@app.delete("/tron-bench-admin/token/{token_id}")
async def revoke_token(
    token_id: str,
    x_tron_bench_admin_token: str = Header(default=""),
) -> JSONResponse:
    if not _admin_authorized(x_tron_bench_admin_token):
        return json_error(401, "unauthorized")
    if not TOKEN_ID_RE.fullmatch(token_id):
        return json_error(404, "token not found")

    with AUTH_LOCK:
        try:
            state = _load_token_state()
        except ValueError:
            return json_error(503, "token state is invalid")
        if state is None or not state["enabled"] or state["token_id"] != token_id:
            return json_error(404, "token not found")
        revoked = dict(state)
        revoked["enabled"] = False
        revoked["token"] = ""
        try:
            _atomic_write(TOKEN_STATE_FILE, json.dumps(revoked, separators=(",", ":")) + "\n")
        except OSError:
            return json_error(503, "token state could not be written")

    return json_response(200, {"ok": True, "token_id": token_id, "revoked": True})


@app.put("/tron-bench-upload/{run_id}/{filename}")
@app.post("/tron-bench-upload/{run_id}/{filename}")
async def upload(
    run_id: str,
    filename: str,
    request: Request,
    x_tron_bench_token: str = Header(default=""),
) -> JSONResponse:
    expected = _current_upload_token()
    if not expected or not hmac.compare_digest(x_tron_bench_token, expected):
        return json_error(401, "unauthorized")
    if not RUN_RE.fullmatch(run_id) or filename not in ALLOWED_FILES:
        return json_error(400, "invalid run id or file name")

    content_length = request.headers.get("content-length")
    try:
        declared_length = int(content_length) if content_length is not None else None
    except ValueError:
        declared_length = -1
    if declared_length is not None and (declared_length < 0 or declared_length > MAX_BYTES):
        return json_error(413, "invalid or oversized body")

    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir = ROOT / run_id
    run_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(ROOT, 0o700)
    os.chmod(run_dir, 0o700)
    target = run_dir / filename

    digest = hashlib.sha256()
    total = 0
    fd, temp_name = tempfile.mkstemp(prefix=f".{filename}.", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BYTES:
                    return json_error(413, "oversized body")
                out.write(chunk)
                digest.update(chunk)
            out.flush()
            os.fchmod(out.fileno(), 0o600)

        new_sha = digest.hexdigest()
        if target.exists():
            old_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if old_sha != new_sha:
                return json_error(409, "file already exists with different content")
            return json_response(
                200,
                {"ok": True, "run_id": run_id, "file": filename, "sha256": old_sha},
            )

        os.replace(temp_name, target)
        temp_name = ""
        return json_response(
            201,
            {"ok": True, "run_id": run_id, "file": filename, "sha256": new_sha},
        )
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
