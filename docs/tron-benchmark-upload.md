# HTTPS benchmark upload

The Windows benchmark can publish its two safe report files to the configured
HTTPS endpoint. It never uploads wallet files, private keys, stdout/stderr
logs, or anything except `summary.txt` and `benchmark.csv`.

Set the token once in the terminal that will run the benchmark:

```powershell
$env:TRON_BENCH_UPLOAD_TOKEN = "<token supplied for this endpoint>"
$env:TRON_BENCH_UPLOAD_URL = "https://turbobuff.beer/tron-bench-upload"
.\bench.cmd
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
