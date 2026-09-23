# Upload only summary.txt and benchmark.csv from one or more report folders.
# Set TRON_BENCH_UPLOAD_TOKEN in the environment; never put the token in a file
# shipped with the release.
param(
    [string[]]$ReportDir,
    [string]$UploadUrl = $(if ($env:TRON_BENCH_UPLOAD_URL) { $env:TRON_BENCH_UPLOAD_URL } else { "https://turbobuff.beer/tron-bench-upload" }),
    [ValidateRange(0, 5)][int]$Retries = 2
)
$ErrorActionPreference = "Stop"
$token = $env:TRON_BENCH_UPLOAD_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) { throw "Set TRON_BENCH_UPLOAD_TOKEN before uploading." }

if (-not $ReportDir) {
    $searchRoot = Split-Path -Parent $PSScriptRoot
    $ReportDir = @(
        Get-ChildItem -LiteralPath $searchRoot -Directory |
            Where-Object { $_.Name -match '^opencl-diagnostic-' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
            ForEach-Object { $_.FullName }
        Get-ChildItem -LiteralPath $searchRoot -Directory |
            Where-Object { $_.Name -match '^vulkan-benchmark-' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
            ForEach-Object { $_.FullName }
    )
}
if (-not $ReportDir) { throw "No opencl-diagnostic-* or vulkan-benchmark-* report folders found." }

foreach ($dir in $ReportDir) {
    $item = Get-Item -LiteralPath $dir
    if (-not $item.PSIsContainer) { throw "Not a report folder: $dir" }
    $runId = $item.Name
    if ($runId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$') { throw "Unsafe report folder name: $runId" }
    foreach ($name in @("summary.txt", "benchmark.csv")) {
        $file = Join-Path $item.FullName $name
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            Write-Warning "Skipping missing $file"
            continue
        }
        $uri = "$($UploadUrl.TrimEnd('/'))/$runId/$name"
        Write-Host "Uploading $file -> $uri"
        $uploaded = $false
        for ($attempt = 0; $attempt -le $Retries -and -not $uploaded; $attempt++) {
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Method Put -Uri $uri `
                    -Headers @{ "X-Tron-Bench-Token" = $token } `
                    -ContentType "application/octet-stream" -InFile $file -TimeoutSec 30
                Write-Host ("Upload response: HTTP {0}; {1}" -f [int]$response.StatusCode, $response.Content)
                $uploaded = $true
            } catch {
                $status = "unknown"
                $body = ""
                if ($_.Exception.Response) {
                    $status = [int]$_.Exception.Response.StatusCode
                    try {
                        $reader = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
                        $body = $reader.ReadToEnd()
                        $reader.Dispose()
                    } catch { }
                }
                if ($attempt -lt $Retries -and ($status -eq "unknown" -or $status -eq 408 -or $status -eq 429 -or $status -ge 500)) {
                    Write-Warning ("Upload attempt {0} failed: HTTP {1}; retrying: {2}" -f ($attempt + 1), $status, $_.Exception.Message)
                    Start-Sleep -Seconds ($attempt + 1)
                } else {
                    throw ("Upload failed for {0}: HTTP {1}; {2}; {3}" -f $file, $status, $body, $_.Exception.Message)
                }
            }
        }
    }
}
Write-Host "Benchmark upload complete. Only summary.txt and benchmark.csv were sent."
