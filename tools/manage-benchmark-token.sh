#!/usr/bin/env bash
# Issue or revoke the benchmark upload token from a Mac with the saved
# codex-server SSH profile. The admin secret is used in memory only.
set -euo pipefail

BASE_URL="${TRON_BENCH_BASE_URL:-https://turbobuff.beer}"
ADMIN_URL="${TRON_BENCH_ADMIN_URL:-${BASE_URL%/}/tron-bench-admin}"
ACTION="${1:-issue}"

if [[ "$ACTION" != "issue" && "$ACTION" != "revoke" ]]; then
    printf 'Usage: %s issue | revoke TOKEN_ID\n' "$0" >&2
    exit 2
fi

if [[ "$ACTION" == "revoke" ]]; then
    TOKEN_ID="${2:-}"
    if [[ ! "$TOKEN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$ ]]; then
        printf 'Invalid token id. Use the token_id returned by issue.\n' >&2
        exit 2
    fi
fi

ADMIN_TOKEN="$(ssh codex-server 'cat /etc/tron-bench-upload/admin-token')"
if [[ -z "$ADMIN_TOKEN" || "$ADMIN_TOKEN" == *$'\n'* || "$ADMIN_TOKEN" == *$'\r'* ]]; then
    unset ADMIN_TOKEN
    printf 'The server admin token is missing or malformed.\n' >&2
    exit 1
fi

headers=(-H "X-Tron-Bench-Admin-Token: $ADMIN_TOKEN" -H 'Accept: application/json')
if [[ "$ACTION" == "issue" ]]; then
    curl --fail --silent --show-error --retry 2 --retry-delay 1 \
        "${headers[@]}" -X POST "$ADMIN_URL/token"
else
    curl --fail --silent --show-error --retry 2 --retry-delay 1 \
        "${headers[@]}" -X DELETE "$ADMIN_URL/token/$TOKEN_ID"
fi
printf '\n'
unset ADMIN_TOKEN headers
