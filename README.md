# TRON Vanity Generator — CPU + OpenCL + CUDA + Metal

[![Windows release](https://img.shields.io/github/v/release/zalupa-ronaldo/tron-vanity-generator-hybrid?display_name=tag)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/releases)
[![Build](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml/badge.svg)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid TRON vanity-address generator with OpenCL on Windows, native CUDA on
Windows/Linux, and native Metal on Apple Silicon. The Windows release embeds
CUDA PTX: NVIDIA users need a compatible driver, without installing the CUDA
Toolkit. AMD GPUs use OpenCL.

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
`--case-sensitive` is used. The OpenCL/CUDA kernel uses a flattened Aho–Corasick
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
.\build\tron_vanity_generator.exe --backend cuda --seconds 60
```

If OpenCL is unavailable, `--backend opencl` falls back to CPU. `auto` uses
CPU plus every available OpenCL GPU. CUDA is explicitly selected with
`--backend cuda`; it does not run alongside OpenCL on the same NVIDIA GPU.

Discrete GPUs default to a larger `2^20` GPU batch to reduce command-queue
gaps. Integrated GPUs stay at `2^16` to keep desktop responsiveness. Use
`--gpu-batch` to override this with a power-of-two value from 1024 to 1048576.

## GPU-resident mode (experimental)

`--gpu-resident` keeps the dictionary, address checking and a device result ring
on the GPU. On OpenCL/CUDA, a lightweight GPU RNG writes one 32-byte scalar per
chunk. The host reads that scalar, expands its public point once with
libsecp256k1, uploads 64 bytes, and the GPU scans the consecutive range. This
96-byte round trip and one CPU point multiplication are amortized over the
whole chunk; after that, the CPU receives only complete matches for independent
secp256k1/address verification and local JSONL writing. Keeping the full
256-bit fixed-base multiplication out of OpenCL avoids an observed Windows
RDNA4 compiler hang. Chunks remain bounded to avoid Windows WDDM timeouts:

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

Metal walks a compile-time batch of consecutive curve points per lane and uses
one modular inverse for the whole batch. The conservative default remains 32,
but the batch can be any power of two through 1,024 as long as
`group-size * keys-per-lane <= 65536`. A balanced, same-temperature profile on
a base 10-GPU-core M4 selected `--gpu-group-size 128
--metal-keys-per-lane 512`: 34.51 M keys/s versus 31.87 M keys/s for the old
256-by-32 configuration (about 8.3% faster). The production worker subsequently
held 33.3-34.1 M keys/s while matching a 1,665-word dictionary.

The large M4 batch is intentional. Xcode Metal System Trace reports 85,168 B
of compiler spill per thread at 512 keys/lane, but that state is amortized over
512 keys and unified-memory bandwidth remains sufficient. Smaller batches lose
more to secp256k1 inversion than they save in spill traffic. A separate run
independently verified 121 generated address/key pairs at 128-by-512, with no
duplicate keys. Rates from upstream prefix-only search are not directly
comparable to full Base58 dictionary matching.

The Metal path also uses a named-lane Keccak-f[1600] permutation. It keeps the
25 lanes in scalar variables and performs Rho/Pi as one in-place cycle instead
of dynamically indexing three temporary arrays. A same-temperature A/B on the
base M4 measured 27.8 M keys/s versus 23.1 M keys/s for the indexed reference
permutation (about 20% faster); cold short runs reached 34.3 M keys/s. Startup
compares both permutations over 1,024 deterministic blocks.

Already-saved dictionary words are removed from the live DFA output tables
between OpenCL, CUDA and Metal dispatches. This preserves the existing
one-result-per-word behavior without sending millions of duplicate short-word
records back to the CPU. The Metal grid is scaled inversely with keys per lane,
keeping command buffers near the same key count as the batch changes; the tuned
M4 command is about 0.76 seconds instead of growing into a multi-second
dispatch.

```bash
./build-mac-metal/tron_vanity_generator --bench-resident --backend metal \
  --words words.example.txt --bench-seconds 2 --gpu-chunk-ms 100 \
  --gpu-group-size 128 --metal-keys-per-lane 512
```

For stage-by-stage diagnosis, use `--metal-profile-stages`; it compiles
separate EC, Keccak, SHA-256, Base58 and dictionary variants and writes no
wallet file. `--metal-profile-indexed-keccak` selects the old Keccak only for
an A/B profile. `--metal-profile-max-threads N` exposes the Metal compiler's
occupancy hint for controlled experiments; it is not enabled in production
because the M4 A/B stayed within run-to-run noise.

The deterministic hardware profiler measures arithmetic, both field
representations, SIMD shuffles, thread-private spill cliffs, threadgroup
capacity and unified-memory access without using wallet data:

```bash
./build-mac-metal/tron_vanity_generator --metal-hw-profile \
  --bench-seconds 0.25 \
  --metal-hw-json benchmarks/apple-m4-metal-hardware-2026-09-21.json
```

Use `--metal-hw-case field|private|memory|threadgroup|group` to select one
category. The full M4 methodology, Xcode spill measurements and conclusions
are in [docs/apple-m4-metal-hardware-profile.md](docs/apple-m4-metal-hardware-profile.md).

Resident work-group size can be tuned per GPU without rebuilding:

```powershell
.\tron_vanity_generator.exe --backend opencl --gpu-resident `
  --gpu-group-size 256 --gpu-rng philox --seconds 60
```

Supported values are `64`, `128` and `256`; `256` is the default. If a driver
reports a compile or launch failure, retry with `128`.

Resident startup prints the lightweight RNG/scan compilation, 32-bit offset
table upload, ring allocation and GPU-CSPRNG/CPU-expansion validation stages
separately. The first launch can spend time in the vendor compiler while it
populates the driver cache; this is CPU-side initialization and therefore does
not show as GPU utilization. The resident build also excludes unrelated RNG
implementations and the legacy/profile/test entry points from each JIT. Every
returned private key/address/dictionary match is recomputed on the CPU before
it can be written. If initialization is still blocked by a vendor driver, the
regular OpenCL path remains available:

```powershell
.\tron_vanity_generator.exe --backend opencl --ec-window 8 `
  --gpu-batch 1048576 --seconds 60
```

### Full benchmark matrix

`--bench` compares every backend available on the current machine. With
`--backend auto` it runs the CPU worker, the legacy OpenCL tuning matrix,
resident OpenCL/CUDA for `chacha12`, `aes-ctr` and `philox`, and resident Metal for
the same three RNGs on Apple Silicon. Explicit `--backend cpu|opencl|cuda|metal`
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

## Native CUDA (experimental)

The CUDA backend uses NVIDIA's Driver API and precompiled PTX, sharing the
resident integer secp256k1, Keccak, SHA-256, Base58 and dictionary code with
OpenCL. CUDA performs the RNG and range scan; libsecp256k1 expands one base
point per chunk on the CPU and independently verifies every returned match.
Device buffers, the CUDA context, module and stream are released when the
backend is destroyed. Contexts are rebound when work moves to a worker thread.

Requirements: NVIDIA compute capability 5.2+ and a driver supporting CUDA 12.6
PTX (or newer). AMD RX 9070 XT and Apple GPUs do not support native CUDA.
CUDA failure is reported explicitly, without silently running on the CPU.

Windows CMD commands (each command is one line):

```bat
tron_vanity_generator.exe --list
tron_vanity_generator.exe --backend cuda --gputest
tron_vanity_generator.exe --backend cuda --words words.txt --seconds 60
tron_vanity_generator.exe --backend cuda --bench-resident --words words.txt --bench-seconds 3
```

CUDA always uses the bounded resident pipeline; `--gpu-resident` is optional.
Use `--gpu-group-size 64|128|256`, `--gpu-chunk-ms 8..100`, and
`--gpu-buffer-mb` to tune it. `--gpu-batch`, `--ec-window`, `--keys-per-item`
and `--mont-n` are legacy OpenCL settings. `--gpu-poll-ms` is retained for CLI
compatibility; each resident dispatch already waits for and drains its results,
so an extra idle delay is no longer inserted. `chacha12` is the default RNG;
the other two modes are for benchmarking, not recommended for funded wallets.

`--backend cuda --gputest` checks 1,024 GPU-produced key/address pairs against
libsecp256k1 per RNG, across two chunks, without saving wallets or printing
private keys. Build-time CTest also executes the shared kernel body on CPU
and checks scalar boundaries, address encoding, ring overflow and removed
dictionary outputs. These tests and successful PTX compilation do not establish
GPU throughput or replace validation on an actual NVIDIA device.

To include CUDA when building on Windows:

```powershell
python -m pip install nvidia-cuda-nvrtc-cu12==12.6.85
powershell -ExecutionPolicy Bypass -File .\build.ps1 -EnableCuda
```

To build on Linux (C++ toolchain, CMake, Python and initialized submodules):

```bash
python3 -m pip install nvidia-cuda-nvrtc-cu12==12.6.85
cmake -S . -B build-cuda -DTRON_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/tron_vanity_generator --backend cuda --gputest
```

Only NVIDIA's NVRTC compiler component is needed at build time, even on a
machine without a GPU. It compiles all three RNG variants into embedded PTX;
the shipped executable loads only the NVIDIA driver. For cross-compilation,
generate the header using `scripts/build_cuda_ptx.py` on Windows/Linux and pass
`-DTRON_CUDA_PTX_HEADER=/absolute/path/cuda_ptx.h` with `TRON_ENABLE_CUDA=ON`.
Without this option, a standard build keeps CUDA disabled and reports that in
`--list`.

Implementation references: [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/)
and [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/).
