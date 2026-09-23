# Exercise the packaged Vulkan benchmark orchestration without a GPU.
$ErrorActionPreference = "Stop"
$root = Join-Path ([IO.Path]::GetTempPath()) ("tron-vulkan-launcher-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $root | Out-Null
$fixture = Join-Path $root "fixture.exe"
Add-Type -OutputAssembly $fixture -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
public class Fixture {
    static string Value(string[] args, string name) {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : "";
    }
    public static int Main(string[] args) {
        bool test = Array.IndexOf(args, "test-vulkan") >= 0;
        string mode = Environment.GetEnvironmentVariable("TRON_VULKAN_FIXTURE");
        string batch = Value(args, "--vulkan-curve-batch");
        string name = test ? "test" : batch == "" ? "opencl" : "vulkan-" + batch;
        File.AppendAllText("calls.txt", name + Environment.NewLine);
        if (test) {
            if (mode == "fail-test") return 2;
            Console.WriteLine("Vulkan resident repeated dispatch + CPU verification PASS (no wallets)");
            return 0;
        }
        if (Value(args, "--words") != "words.txt" || Value(args, "--bench-seconds") != "1" ||
            Array.IndexOf(args, "--no-config") < 0) return 3;
        if (mode == "fail-profile" && batch == "4") return 4;
        if (batch == "") Console.WriteLine("wall 1.000 s, wall speed 100.000 M/s");
        else Console.WriteLine("wall: 25000000 keys / 1.000 s, " +
                               (batch == "4" ? "30.00" : "20.00") + " M/s, 10 dispatches");
        return 0;
    }
}
'@
$previousMode = $env:TRON_VULKAN_FIXTURE
try {
    foreach ($case in @(
        @{ Mode = "success"; Exit = 0; Calls = "test,opencl,vulkan-1,vulkan-4,vulkan-1,vulkan-4,opencl" },
        @{ Mode = "fail-test"; Exit = 1; Calls = "test" },
        @{ Mode = "fail-profile"; Exit = 1; Calls = "test,opencl,vulkan-1,vulkan-4,vulkan-1,vulkan-4,opencl" }
    )) {
        $dir = Join-Path $root ("test " + $case.Mode)
        New-Item -ItemType Directory -Path $dir | Out-Null
        Copy-Item -LiteralPath $fixture -Destination (Join-Path $dir "tron_vanity_generator.exe")
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "../bench-vulkan.ps1") -Destination $dir
        Set-Content -LiteralPath (Join-Path $dir "words.txt") -Value "energy" -Encoding ASCII
        $env:TRON_VULKAN_FIXTURE = $case.Mode
        $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir "bench-vulkan.ps1") -Seconds 1 2>&1
        if ($LASTEXITCODE -ne $case.Exit) {
            throw "$($case.Mode) exit mismatch: $LASTEXITCODE`n$($output -join "`n")"
        }
        $calls = (Get-Content -LiteralPath (Join-Path $dir "calls.txt")) -join ","
        if ($calls -ne $case.Calls) { throw "$($case.Mode) call order mismatch: $calls" }
        $reports = @(Get-ChildItem -LiteralPath $dir -Recurse -Filter summary.txt)
        if ($reports.Count -ne 1) { throw "$($case.Mode) missing summary" }
        $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        if ($case.Mode -eq "fail-test") {
            if (-not $report.Contains("Vulkan correctness failed")) { throw "Missing fail-closed report" }
        } else {
            $csv = Import-Csv -LiteralPath (Join-Path $reports[0].Directory.FullName "benchmark.csv")
            if ($csv.Count -ne 7) { throw "$($case.Mode) expected seven benchmark rows" }
            $vulkan4 = @($csv | Where-Object { $_.Test -eq "04-vulkan-4-A" })[0]
            if ($case.Mode -eq "success") {
                if ($vulkan4.Result -ne "PASS" -or $vulkan4.MKeysPerSecond -ne "30") {
                    throw "Vulkan batch-4 wall rate was not parsed"
                }
            } elseif ($vulkan4.Result -notlike "FAIL*") { throw "Profile failure was not recorded" }
        }
        if (Get-ChildItem -LiteralPath $dir -Recurse -Filter *.jsonl) {
            throw "$($case.Mode) wrote wallet output"
        }
    }
    Write-Host "Vulkan benchmark launcher PASS (success and failure paths)"
} finally {
    $env:TRON_VULKAN_FIXTURE = $previousMode
    if (Test-Path -LiteralPath $root) { [IO.Directory]::Delete($root, $true) }
}
# The final fixture intentionally exits 1. Do not leak that LASTEXITCODE into
# the GitHub Actions PowerShell host after all assertions have passed.
exit 0
