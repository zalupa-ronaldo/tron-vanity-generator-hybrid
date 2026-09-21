# TRON Vanity Generator — CPU + OpenCL

[![Windows release](https://img.shields.io/github/v/release/zalupa-ronaldo/tron-vanity-generator-hybrid?display_name=tag)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/releases)
[![Build](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml/badge.svg)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid TRON vanity-address generator for Windows. It runs the CPU worker and
all detected OpenCL GPUs concurrently. OpenCL is used for both AMD and NVIDIA
cards, so version 1 does not require the CUDA Toolkit. A native CUDA backend is
reserved for version 2.

The GPU architecture is based on and attributes
[hlzzhqb/tron-vanity-generator](https://github.com/hlzzhqb/tron-vanity-generator)
(MIT). The project also bundles Bitcoin Core `libsecp256k1` as an MIT
submodule; see `NOTICE` and `third_party/secp256k1/COPYING`.

The Metal resident point walk and per-lane batch inversion were inspired by
[mrtozner/tron-vanity-metal](https://github.com/mrtozner/tron-vanity-metal)
(MIT). Our kernel keeps GPU-generated starting scalars and checks the full
dictionary against each Base58 address; benchmark rates from the two projects
therefore measure different workloads and hardware.

## What it searches

`--words` supplies a Base58-compatible dictionary. Every address is scanned
after its initial `T`; lowercase and uppercase variants are matched unless
`--case-sensitive` is used. The OpenCL kernel uses a flattened Aho–Corasick
automaton, so CPU and GPU use the same matching semantics.

`words.example.txt` is a curated starter dictionary with 350+ entries across
TRON/energy, payments, names, technology, gaming, space and memorable phrases.
Copy it to `words.txt` and add your own Base58-compatible tokens. The matcher
accepts words anywhere in the address, including the final characters; longer
matches are especially rare and useful.

## Build on Windows

Install Visual Studio Build Tools with the C++ workload, CMake, Ninja and Git.
The GPU path only needs the OpenCL runtime shipped with the AMD or NVIDIA
driver.

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\build\tron_vanity_generator.exe --selftest
.\build\tron_vanity_generator.exe --list
```

## Run hybrid mode

```powershell
.\build\tron_vanity_generator.exe `
  --seconds 259200 `
  --threads 16 `
  --words ..\tron-vanity\words.txt `
  --out results
```

The console progress line shows elapsed percentage, CPU threads and keys/s,
GPU keys/s, total throughput and match count. `--seconds 0` runs until Ctrl+C.

Useful checks:

```powershell
.\build\tron_vanity_generator.exe --gputest
.\build\tron_vanity_generator.exe --hashtest
.\build\tron_vanity_generator.exe --matchtest
.\build\tron_vanity_generator.exe --backend cpu --threads 16 --seconds 60
.\build\tron_vanity_generator.exe --backend opencl --seconds 60
.\build\tron_vanity_generator.exe --backend opencl --gpu-batch 1048576 --seconds 60
.\build\tron_vanity_generator.exe --backend opencl --gpu-resident --gpu-rng chacha12 --seconds 60
```

If OpenCL is unavailable, `--backend opencl` falls back to CPU. `auto` uses
CPU plus every available OpenCL GPU.

Discrete GPUs default to a larger `2^20` GPU batch to reduce command-queue
gaps. Integrated GPUs stay at `2^16` to keep desktop responsiveness. Use
`--gpu-batch` to override this with a power-of-two value from 1024 to 1048576.

## GPU-resident mode (experimental)

`--gpu-resident` keeps the dictionary, 256-bit GPU-generated scalars, address
checking and a device result ring on the GPU. The CPU receives only complete
matches for local JSONL writing. Chunks are bounded to avoid Windows WDDM
timeouts:

```powershell
.\build\tron_vanity_generator.exe --backend opencl --gpu-resident `
  --gpu-rng chacha12 --gpu-buffer-mb 128 --gpu-chunk-ms 32 --gpu-poll-ms 50 `
  --seconds 60
```

`chacha12` is the production resident RNG. `philox` and `aes-ctr` are available
for cross-vendor benchmarking; all three use a per-run OS-generated seed and
rejection sampling against the secp256k1 order. If the device result ring
overflows, the run stops rather than silently dropping a wallet result.

The Metal resident backend is available on Apple Silicon with macOS 26+ and
Xcode 26.6's Metal Toolchain:

```bash
sudo xcode-select --switch /Applications/Xcode-26.6.0.app/Contents/Developer
./build-macos.sh
./build-mac-metal/tron_vanity_generator --backend metal --gpu-resident \
  --gpu-rng chacha12 --gpu-buffer-mb 128 --gpu-chunk-ms 32 \
  --seconds 60
```

The MSL kernel is compiled and linked as a Metal library during the macOS
build. The current release keeps Metal opt-in; `--backend auto` will select it
after the cross-validation acceptance run.

Metal processes 32 consecutive curve points per lane and uses one modular
inverse for the batch. Tune with `--metal-keys-per-lane 4|8|16|32` if a
different Apple GPU prefers a smaller batch. On a base M4 with the same
dictionary and 256-lane groups, `chacha12` measured 14.3 M keys/s at 4,
18.7 M at 8, 22.0 M at 16, and 23.5 M at 32 (2026-09-21, 2-second
benchmark passes). The 32-key mode also passed independent verification of
121 generated address/key pairs. Rates from upstream prefix-only search are
not directly comparable to full Base58 dictionary matching.

The Metal path also uses a named-lane Keccak-f[1600] permutation. It keeps the
25 lanes in scalar variables and performs Rho/Pi as one in-place cycle instead
of dynamically indexing three temporary arrays. A same-temperature A/B on the
base M4 measured 27.8 M keys/s versus 23.1 M keys/s for the indexed reference
permutation (about 20% faster); cold short runs reached 34.3 M keys/s. Startup
compares both permutations over 1,024 deterministic blocks.

Already-saved dictionary words are removed from the live DFA output tables
between Metal dispatches. This preserves the existing one-result-per-word
behavior without sending millions of duplicate short-word records back to the
CPU. In a sustained test the production path reached 27.1 M keys/s against a
27.9 M keys/s GPU-only ceiling. An independent run recomputed 142 generated
addresses from their keys and verified every dictionary match with no duplicate
words.

```bash
./build-mac-metal/tron_vanity_generator --bench-resident --backend metal \
  --words words.example.txt --bench-seconds 2 --gpu-chunk-ms 100 \
  --gpu-group-size 256 --metal-keys-per-lane 32
```

For stage-by-stage diagnosis, use `--metal-profile-stages`; it compiles
separate EC, Keccak, SHA-256, Base58 and dictionary variants and writes no
wallet file. `--metal-profile-indexed-keccak` selects the old Keccak only for
an A/B profile.

Resident work-group size can be tuned per GPU without rebuilding:

```powershell
.\tron_vanity_generator.exe --backend opencl --gpu-resident `
  --gpu-group-size 256 --gpu-rng philox --seconds 60
```

Supported values are `64`, `128` and `256`; `256` is the default. If a driver
reports a compile or launch failure, retry with `128`.

### Full benchmark matrix

`--bench` compares every backend available on the current machine. With
`--backend auto` it runs the CPU worker, the legacy OpenCL tuning matrix,
resident OpenCL for `chacha12`, `aes-ctr` and `philox`, and resident Metal for
the same three RNGs on Apple Silicon. Explicit `--backend cpu|opencl|metal`
restricts the matrix to that backend. The legacy OpenCL section also scans
EC window, Montgomery batch size, keys-per-item, match length and GPU batch
size.

On Windows, use `--bench-resident` to skip the legacy tuning matrix and go
straight to the resident OpenCL RNG comparison (recommended for a quick
9070 XT check):

```powershell
.\tron_vanity_generator.exe --backend opencl --bench-resident `
  --words words.txt --bench-seconds 15 --gpu-chunk-ms 100 --gpu-poll-ms 1000
```

Use `--bench-seconds` to control the timed duration of each row (default 1
second; the legacy tuning matrix has additional warm-up passes):

```bash
./build-mac-metal/tron_vanity_generator --bench --backend auto \
  --words words.example.txt --bench-seconds 1
```

The benchmark never writes wallet JSONL or private keys. It reports
unavailable backends and continues with the methods that the machine supports.

## Results and security

Runtime output is JSONL with `address`, `words` and `private_key`. Private
keys are written only to the local result file and are never printed. Treat
that file as secret material, do not commit it, and do not upload it to a
public repository. The repository `.gitignore` excludes results, logs, PID
files and generated address tables.

The release archive contains only the executable, a starter dictionary and
the project documentation. It does not contain wallet exports, private keys,
local logs or benchmark output.

## CUDA roadmap

CUDA is intentionally not required for v1: NVIDIA users get a no-toolkit path
through OpenCL. Version 2 can add a native CUDA backend behind the same
`Backend` interface without changing the dictionary, result format or CPU
fallback.
