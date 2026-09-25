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
- `tron-vanity.conf`: measured RX 9070 XT Vulkan defaults. `src/run_config.cpp`:
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

## Verified Vulkan decision, 2026-09-24

The RX 9070 XT passed the no-wallet Vulkan correctness gate and measured
123.18 M keys/s at curve 1 / affine 4 / resident group 8, 126.09 M/s at the
same math with group 16, 141.79 M/s at curve 4 / affine 4 / group 16, and
157.14 M/s at curve 4 / affine 8 / group 16. These are user-provided resident
benchmark rates using the 10x26 field and 131072-key submits. The latter is
the selected explicit-Vulkan default and is about 50% above the earlier
104.8 M/s OpenCL reference. A 262144-key submit with group 16 failed at
`vkAllocateMemory -2`, so it is an allocation limit, not evidence that the
curve or address math is incorrect. The redacted measurements are in
[`amd-rx9070xt-vulkan-2026-09-24.json`](../benchmarks/amd-rx9070xt-vulkan-2026-09-24.json).
The bundled `tron-vanity.conf` now selects the measured resident Vulkan winner
(curve 4, affine 8, 131072-key submits, group 16, 10x26). `bench.cmd` compares
the full-address OpenCL and Vulkan rows and can switch the config back to
OpenCL only when a later valid measurement wins. Wallet-output validation
remains a separate gate.

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
4. Vulkan: `vulkan/curve.comp`, `vulkan/keccak.comp`, `vulkan/checksum.comp`,
   `vulkan/base58.comp`, `vulkan/match.comp` and the native backend chain a
   10x26 secp256k1 offset walk, full-address encoding and a bounded atomic
   match ring. The tests compare every stage and ring record with
   CPU/libsecp256k1, including offset-window and capacity boundaries. The
   Windows release compiles this backend, and its adjacent config selects the
   measured Vulkan winner. `--backend vulkan` fails closed and
   `test-vulkan` exercises the production backend without writing wallets,
   including ring, rollover and synthetic device-loss checks.
   The current source separates projective curve output from a GPU-only affine
   stage. `--vulkan-affine-batch 4|8` selects one inversion per four or eight
   points, and `--vulkan-curve-batch 1|4` selects the projective walk. The RX
   correctness gate and matched full-address benchmark now pass. The explicit
   Vulkan defaults are curve 4, affine 8, batch 131072 and resident group 16.
   The normal Windows ZIP and CI artifact package `bench-vulkan.cmd`; its
   reports contain rates and dictionary hashes, not wallets. Compare wall rate
   first and retain CPU verification. A long-running wallet-output run and
   driver-loss recovery still need validation before recommending funds use.
   The byte/word contract and remaining stage gates are in
   [`vulkan-stage-contract.md`](vulkan-stage-contract.md).

   The Vulkan build also carries an opt-in `--vulkan-field 8x32` variant.
   It uses eight 32-bit field limbs and a 96-byte-per-key Jacobian scratch
   layout. The RX field8 self-test passed, but its measured no-wallet profile
   was only 17.77 M/s wall (21.97 M/s GPU-stage rate), so the validated 10x26
   path remains the default.

   All per-key Vulkan stage buffers are in a separate GPU-only allocation;
   the RX profile reports host-visible device-local memory. Resident grouping
   records six GPU stage dispatches (curve, affine, Keccak, checksum, Base58
   and match) before one fence, rather than fencing every logical batch. A
   262144-key submit with group 16 failed at `vkAllocateMemory -2`; do not
   promote that size without a memory-layout change. If future profiles show
   another limiter, measure fence wait, ring collection and command recording
   separately, preserve the same CPU verification, and compare full wall rate.

## Reproducibility and rollout

Build/CTest locally, run Windows launcher fixture in CI, then tag a release
only after Windows and Linux jobs pass. The Windows release must contain the
exe, `tron-vanity.conf`, `words.txt` starter dictionary, and `bench.cmd`.
The recommended RX config remains GPU-only and `seconds=0` (until Ctrl+C).
It fails if OpenCL is missing instead of silently falling back to CPU. Keep
untested optimizations opt-in. Benchmark reports should note dictionary size,
driver, compiler flags, group, batches, repeated wall rates and GPU-stage
times; never recommend a setting from a single noisy run.
