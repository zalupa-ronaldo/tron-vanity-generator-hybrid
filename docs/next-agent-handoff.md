# Engineering handoff: resident GPU search

This document is for follow-on agents. Treat the current checkout, CI and
fresh benchmarks as authoritative. Never paste, commit, upload or print
private keys or `wallets-*.jsonl`. A benchmark JSON or `summary.txt` should
contain rates and public addresses only.

## End-to-end invariants

The only comparable throughput is full TRON-address processing: secp256k1
point step, Keccak, double SHA-256 checksum, Base58Check encoding, full
dictionary search and result-ring accounting. Prefix-only or kernel-event
rates must not be compared to full wall throughput. Every math or layout
change must pass a GPU/CPU scalar and address self-test, including base-window
rollover and ring wrap/overflow paths. Overflow must fail closed instead of
dropping matches. A successful benchmark never writes wallet files.

## Where to work

- `kernels/secp256k1_tron.cl`: six isolated staged OpenCL kernels and
  compile-time options. `src/resident_backend.cpp`: stage dispatch, scratch
  buffers, result ring and wall/event timings. `src/ocl.cpp`: dynamic OpenCL
  loader and queue calls.
- `kernels/tron_vanity_resident.metal`, `src/metal_backend.mm`:
  M4 resident point walk and profiling. Detailed controlled measurements:
  [`apple-m4-metal-hardware-profile.md`](apple-m4-metal-hardware-profile.md).
- `bench.cmd`: one-click orchestration of the bounded OpenCL variant matrix
  (`test-opencl.ps1`) and Vulkan/OpenCL A/B (`bench-vulkan.ps1`). Each suite
  has a separate report folder and correctness gate; both reports are needed.
  `tests/opencl_launcher_test.ps1`: fake-process launcher
  regression test. `tests/opencl_runtime_test.cpp`: actual OpenCL runtime
  integration (CPU OpenCL in Linux CI is correctness evidence, not RX speed).
- `tron-vanity.conf`: conservative RX 9070 XT defaults. `src/run_config.cpp`:
  strict key=value parser; relative dictionary/output paths resolve next to
  the config. The config has no secret values.

## Verified decisions, 2026-09-22

RX 9070 XT staged OpenCL at group 64, curve batch 2, paired affine batch 4,
358-word dictionary measured about 105 M keys/s. Affine batch 2 loses about
20%; batch 8 was about 3% below batch 4. See
[`rx9070xt-opencl-research.md`](rx9070xt-opencl-research.md). The queued
metadata-read mode, SHA ring, curve batches 4/8 and group 128/256 remain
unproven on RX until fresh A/B results arrive. Do not silently promote them.
The 105 M/s profile used an 8 MiB result ring, while the shipped config uses
128 MiB for capacity. Its relative wall rate is unmeasured; `bench.cmd`
now tests 8/128/8 and chunk targets 16/32/64 ms before anyone should tune
the release config on throughput grounds.
The later v1.9.3 rerun, with matching executable and dictionary SHA-256
fingerprints, measured 104.532 M/s with all stage optimizations and 77.799 M/s
with the optimization mask cleared; its partial mask sweep is recorded in
[`amd-rx9070xt-opencl-2026-09-23-partial.json`](../benchmarks/amd-rx9070xt-opencl-2026-09-23-partial.json).
The pasted report ended before the remaining matrix and Vulkan suite, so do
not infer 128 MiB or Vulkan performance from it.

On base M4 with a 1,665-word dictionary, measured Metal 10x26 field
multiplication was 2.13x the 8x32 port; `128 threads x 512 keys/lane` was
8.3% faster than the former `256 x 32` setup. The production command was
already 99.8-99.9% GPU busy, so host double buffering did not address the
measured bottleneck. The private point arrays spill substantially, but the
larger batch still wins by amortizing field inversion. Do not replace 10x26,
reduce the batch blindly, or infer VGPR occupancy from logical array size.

## Highest-value next experiments

1. RX: obtain `bench.cmd` report, then remeasure top variants in A/B/A order.
   Analyze *wall* gain first, stage ns/key second. Use AMD GPA/RGP if possible
   to distinguish VGPR, scratch and ALU limits in affine and curve kernels.
2. RX affine: specialize fixed-generator point addition or use a coalesced
   scratch layout while holding batch=4. Verify each scalar/address; inspect
   VGPR/scratch and reject if wall speed regresses even when instruction count
   improves.
3. M4: prototype struct-of-arrays device scratch for point/prefix state,
   versus SIMD32 cooperative batch inversion. Hold total keys/command fixed;
   measure complete wall and GPU-command times in forward/reverse order after
   stopping only a *test* worker under explicit user control. Never trace or
   publish working secret buffers.
4. Vulkan: `vulkan/probe.comp`, `vulkan/vulkan_probe.cpp` and
   `src/vulkan_probe.h` implement a deterministic native compute dispatch.
   `vulkan/curve.comp`, `vulkan/keccak.comp`, `vulkan/checksum.comp`,
   `vulkan/base58.comp`, `vulkan/match.comp` and `vulkan/vulkan_keccak.cpp`
   chain a native 10x26 secp256k1 offset walk, full-address encoding and a
   bounded atomic match ring. The test compares every stage and ring record
   with CPU/libsecp256k1, including offset-window and capacity boundaries. They
   are built only with `-DTRON_ENABLE_VULKAN=ON`. Linux CI uses Mesa's
   software Vulkan driver; Windows SDK CI compiles the optional stages but
   has no RX 9070 XT to execute them. The Windows release compiles the Vulkan
   backend, but its adjacent config selects OpenCL; Vulkan is explicit opt-in.
   `vulkan/vulkan_backend.cpp` now adds
   reusable bounded dispatch, CSPRNG base rollover, ring drain and CPU
   verification of candidate private keys. `--backend vulkan` is explicit and
   fails closed; `test-vulkan` also exercises the production backend without
   writing wallets, including a full 32,768-slot ring and CSPRNG base
   rollover. The embedded 58-word Base58 alphabet guarantees one match per
   address in that test. The current prototype separates projective curve
   output from a GPU-only affine stage; `--vulkan-affine-batch 4|8` selects
   one inversion per four or eight points, while `--vulkan-curve-batch 1|4`
   selects the projective curve walk. The new variants compile and validate
   as SPIR-V locally, but need the RX correctness gate and a matched profile.
   The earlier RX profile (OpenCL 104.635 M/s, Vulkan 2.885/2.985 M/s) predates
   the split and must not be used as its performance result. The RX reports
   host-visible device-local memory, while about 4.2-4.4 seconds of each
   five-second pre-split Vulkan run was outside GPU stages. The native backend
   is therefore still not performance-ready;
   do not recommend it for funds until a long-running wallet-output test is
   independently verified. The normal Windows ZIP and opt-in CI
   artifact both package `bench-vulkan.cmd`, which first
   runs `test-vulkan` and then measures OpenCL / Vulkan batch 1 / batch 4 /
   batch 1 / batch 4 / OpenCL with bounded child processes and the same
   adjacent `words.txt`. Its `summary.txt` and `benchmark.csv` are safe to
   share; wallets are not produced. The underlying `--vulkan-profile
   --vulkan-curve-batch 1|4 --vulkan-affine-batch 4|8 --bench-seconds 5`
   reports wall rate and per-stage
   GPU timestamps (when supported), plus host-visible memory locality.
   Compare wall rate first. A non-device-local mapping may be limited by PCIe;
   test device-local scratch plus staging before optimizing shader math.
   Synthetic `VK_ERROR_DEVICE_LOST` at queue submit is now tested fail-closed;
   real driver loss/timeouts and long-running RX wallet-output validation still
   need testing. OpenCL running on `clvk` would be a compatibility experiment,
   not evidence of a native Vulkan backend. Khronos's
   [compute guide](https://docs.vulkan.org/guide/latest/compute_shaders.html)
   and the [clspv OpenCL-C mapping](https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md)
   identify API and compiler constraints. Start with deterministic seed/curve
   and address stages, then integrate one stage at a time; do not report a
   Vulkan speed until full-address/dictionary processing is verified.
   The byte/word contract and remaining stage gates are in
   [`vulkan-stage-contract.md`](vulkan-stage-contract.md).

   The measured Vulkan bottleneck is now concrete: the RX completed only about
   1.3 ms of GPU stages per 32,768-key dispatch, while the wall profile spent
   about 4.2-4.4 s outside those stages across roughly 440-457 dispatches in
   five seconds. The resident command-buffer group is now eight logical
   dispatches, so the next RX benchmark must measure whether the reduced fence
   count moves wall rate toward the GPU-stage rate. The Jacobian scratch is now
   a separate GPU-only allocation; verify the profile reports it as
   device-local on the RX. If it does not, prototype a larger bounded batch
   (for example 64K, 128K and 256K keys) with a
   correspondingly sized ring, and separately
   measure fence wait, mapped-buffer/ring drain, address reconstruction and
   command-recording time. Keep the same full-address CPU verification and
   compare wall rate in A/B/A; do not infer a win from GPU timestamps alone.
   The RX already reports host-visible device-local memory, so a staging-buffer
   rewrite is lower priority until a larger-batch test shows that memory
   traffic, rather than per-dispatch synchronization, dominates.

## Reproducibility and rollout

Build/CTest locally, run Windows launcher fixture in CI, then tag a release
only after Windows and Linux jobs pass. The Windows release must contain the
exe, `tron-vanity.conf`, `words.txt` starter dictionary, and `bench.cmd`.
The recommended RX config remains GPU-only and `seconds=0` (until Ctrl+C).
It fails if OpenCL is missing instead of silently falling back to CPU. Keep
untested optimizations opt-in. Benchmark reports should note dictionary size,
driver, compiler flags, group, batches, repeated wall rates and GPU-stage
times; never recommend a setting from a single noisy run.
