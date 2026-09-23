# HTTPS benchmark upload

The Windows benchmark can publish its two safe report files to the configured
HTTPS endpoint. It never uploads wallet files, private keys, stdout/stderr
logs, or anything except `summary.txt` and `benchmark.csv`.

Create an upload token from a Mac that has the saved `codex-server` SSH
profile. The admin secret is retrieved over SSH and passed to the HTTPS admin
endpoint in memory; it is not printed or saved by the helper:

```bash
./tools/manage-benchmark-token.sh issue
```

The response contains `token_id` and the new upload `token` exactly once.
Copy only the token value into the Windows PowerShell session:

```powershell
$env:TRON_BENCH_UPLOAD_TOKEN = "<token supplied for this endpoint>"
$env:TRON_BENCH_UPLOAD_URL = "https://turbobuff.beer/tron-bench-upload"
.\bench.cmd
```

Issuing a new token immediately invalidates the previous one. To revoke the
current token without issuing a replacement, run this on the Mac:

```bash
./tools/manage-benchmark-token.sh revoke <token_id>
```

`bench.cmd` uploads the newest OpenCL and Vulkan report directories after the
two suites finish. To upload a specific folder instead:

```powershell
.\tools\upload-benchmark-results.cmd .\opencl-diagnostic-20260923-193843-8070b1
```

The server stores reports below `/srv/tron-benchmark-reports/<run-id>/` with
mode `0700`; individual files are `0600`. Uploads require the token in the
`X-Tron-Bench-Token` header, accept only the two filenames above, cap each
file at 2 MiB, and reject conflicting replacement uploads.

The admin API is separate from uploads: `POST /tron-bench-admin/token` issues
a replacement token and `DELETE /tron-bench-admin/token/<token_id>` revokes the
current token. Both require `X-Tron-Bench-Admin-Token`; all JSON responses are
marked `Cache-Control: no-store`.
