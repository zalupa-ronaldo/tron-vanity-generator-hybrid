#!/usr/bin/env bash
set -eu

install -m 0644 /root/tron-nginx-fragment.conf /etc/nginx/snippets/tron-bench-upload.location.conf
cp -a /etc/nginx/sites-available/loveyour.mom /etc/nginx/sites-available/loveyour.mom.bak-tron-bench

install -d -m 0700 -o tronbench -g tronbench /etc/tron-bench-upload
if [ ! -s /etc/tron-bench-upload/admin-token ]; then
    umask 077
    python3 -c 'import os, pathlib, pwd, secrets; p=pathlib.Path("/etc/tron-bench-upload/admin-token"); p.write_text(secrets.token_urlsafe(48)+"\n", encoding="utf-8"); os.chmod(p, 0o600); u=pwd.getpwnam("tronbench"); os.chown(p, u.pw_uid, u.pw_gid)'
fi

python3 - <<'PY'
from pathlib import Path

p = Path('/etc/nginx/sites-available/loveyour.mom')
s = p.read_text()
marker = '''server_name turbobuff.beer www.turbobuff.beer;
    server_tokens off;

    ssl_certificate /etc/letsencrypt/live/turbobuff.beer/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/turbobuff.beer/privkey.pem;
'''
include = '    include /etc/nginx/snippets/tron-bench-upload.location.conf;\n'
if include not in s:
    if s.count(marker) != 1:
        raise SystemExit(f'expected one HTTPS TurboBuff marker, got {s.count(marker)}')
    p.write_text(s.replace(marker, marker + include, 1))
PY

nginx -t
systemctl daemon-reload
chmod 0644 /opt/tron-bench-upload/tron_bench_upload.py
systemctl enable tron-bench-upload.service
systemctl restart tron-bench-upload.service
systemctl is-active tron-bench-upload.service
curl --fail --silent http://127.0.0.1:8768/tron-bench-upload/health
nginx -s reload
