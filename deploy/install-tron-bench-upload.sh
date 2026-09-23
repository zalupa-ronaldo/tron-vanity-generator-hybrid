#!/usr/bin/env bash
set -eu

install -m 0644 /root/tron-nginx-fragment.conf /etc/nginx/snippets/tron-bench-upload.location.conf
cp -a /etc/nginx/sites-available/loveyour.mom /etc/nginx/sites-available/loveyour.mom.bak-tron-bench

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
systemctl enable --now tron-bench-upload.service
systemctl is-active tron-bench-upload.service
curl --fail --silent http://127.0.0.1:8768/tron-bench-upload/health
nginx -s reload
