#!/usr/bin/env python3
"""Authenticated FastAPI receiver for no-wallet benchmark reports."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse


ROOT = Path(os.environ.get("TRON_BENCH_ROOT", "/srv/tron-benchmark-reports"))
TOKEN_FILE = Path(os.environ.get("TRON_BENCH_TOKEN_FILE", "/etc/tron-bench-upload/token"))
MAX_BYTES = 2 * 1024 * 1024
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
ALLOWED_FILES = {"summary.txt", "benchmark.csv"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": message})


@app.get("/tron-bench-upload/health")
async def health() -> dict[str, object]:
    return {"ok": True, "service": "tron-bench-upload"}


@app.put("/tron-bench-upload/{run_id}/{filename}")
@app.post("/tron-bench-upload/{run_id}/{filename}")
async def upload(
    run_id: str,
    filename: str,
    request: Request,
    x_tron_bench_token: str = Header(default=""),
) -> JSONResponse:
    expected = TOKEN_FILE.read_text(encoding="utf-8").strip()
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
            return JSONResponse(status_code=200, content={"ok": True, "run_id": run_id, "file": filename, "sha256": old_sha})

        os.replace(temp_name, target)
        temp_name = ""
        return JSONResponse(status_code=201, content={"ok": True, "run_id": run_id, "file": filename, "sha256": new_sha})
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
