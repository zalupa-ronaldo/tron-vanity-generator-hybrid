# Native Vulkan pipeline: stage contract and verification gates

The native Vulkan wallet backend is **experimental**. Source builds with
`-DTRON_ENABLE_VULKAN=ON` have a generic dispatch probe, a native secp256k1
10x26 field-math point stage (plus an opt-in 8x32 A/B variant), a separate batched-affine stage, Keccak-256,
SHA-256d, Base58Check and flattened dictionary matching. The curve stage can
compute `P + G` or walk from one base point through three 8-bit offset-table
windows and leaves Jacobian points in GPU-only scratch. The affine stage then
shares one Fermat inversion across 4 or 8 points (`--vulkan-affine-batch`).
The earlier RX profile (2.9-3.0 M keys/s) predates this split and is not a
measurement of the new pipeline. The RX 9070 XT then passed the no-wallet
correctness gate and measured 157.14 M keys/s with curve batch 4, affine batch
8, batch 131072 and resident group 16. This is a user-provided single-run
benchmark, not a wallet-output throughput claim. The generic and deterministic
stage tests do not save wallets. Vulkan requires an explicit backend selection.

## Verified interface to preserve

`vulkan/curve.comp` reads a `3 × 256 × 16`-word table at binding 7. In the
production projective variant, mode 1 reads the one affine base point from
push constants and computes `P0 + offset·G`, where the offset is a
push-constant base plus the invocation ID; it writes 30 words per Jacobian
point `(X,Y,Z)` to binding 5 in field-plane (struct-of-arrays) order for
10x26, or 24 words for the opt-in 8x32 path.
Deterministic test mode 0 uses the per-key
affine points at binding 6 and retains the direct affine output contract for
the stage-test pipeline. The host derives every expected point independently
with `libsecp256k1`; tested offsets cross 255/256, 65535/65536 and end at
`2^22 - 1`.
The production projective batch-4 variant builds the first point from the
offset table and walks `+G` for the remaining three points, avoiding redundant
table lookups and mixed additions. A separate affine variant reads binding 5,
performs one batch inversion for 4 or 8 Z coordinates, reloads X/Y only while
emitting affine public keys to binding 0, and handles a final partial group.
Compute barriers separate curve→affine→Keccak. The host dispatches
`ceil(count/curveBatch)` curve invocations, `ceil(count/affineBatch)` affine
invocations, and `count` address/match invocations.

`vulkan/keccak.comp` takes 16 little-endian `uint32_t` words per 64-byte
uncompressed public key (`X||Y`, no `0x04` prefix) at storage binding 0. It
outputs six little-endian words at binding 1: byte `0x41`, the last 20 bytes
of Keccak-256, and three zero padding bytes. The first push-constant word is
the key count. The test uses up to 257 deterministic, test-only secp256k1
scalars. It dispatches counts 1, 8, 63, 64, 65 and 257, comparing every active
output word with CPU references and checking that inactive workgroup lanes
leave canary-filled output records untouched. `vulkan/checksum.comp` then
reads binding 1 and writes seven little-endian words at binding 2: the 25-byte address plus three zero
padding bytes. A compute-to-compute barrier separates the two dispatches;
both intermediate and final buffers are checked. The test never prints or
saves a scalar.
`vulkan/base58.comp` reads binding 2 and writes nine little-endian words at
binding 3 (34 ASCII characters and two zero padding bytes). The test compares
this final text with `tronAddressFromPubXY()` on every vector. It uses the
same two-digit (base 58²) long division as the OpenCL resident kernel.
`vulkan/match.comp` reads text at binding 3, the immutable DFA/offset table at
binding 10 and the mutable output lists at binding 4. In test mode it writes one 18-word record per address at binding 5
(count, overflow, 16 distinct word IDs). In ring mode it atomically reserves
a slot in metadata binding 8 and writes a 29-word record at binding 9 (key
index, count, flags, reserved, 16 IDs and nine packed words of the GPU
Base58 address). Tests use capacity 8 plus two canary guard slots, compare
concurrent records without assuming write order, cover exact-capacity and
over-capacity cases, and check both overflow flags. The
host constructs the 30 embedded one-letter stage-test words through the production
Dictionary builder (the same list is in `tests/vulkan_words.txt`)
and compares records with `Dictionary::matchIds()`. This is still a *test*
ring: no scalar/address is emitted to user output. Capacity overflow is
fail-closed; a truncated ID list can be recovered only if the production host
recomputes every match from the full CPU dictionary.
The production-backend self-test instead uses the full 58-character Base58
alphabet and starts one full 32,768-key dispatch at the
end of the 22-bit offset window, then forces a one-key dispatch after base
rollover. With every Base58 character in the test dictionary, every address
must produce exactly one ring record. The host validates all 32,769 reported
keys and addresses through the normal production callback without saving
wallets, checks that the base changed, and checks the new offset is one.
After both curve variants pass, it injects a synthetic `VK_ERROR_DEVICE_LOST`
at `vkQueueSubmit` for one bounded production run. The run must report zero
keys, set its stop flag, and reject a retry; the engine then avoids
`vkDeviceWaitIdle` on the abandoned device. This is fault-path validation,
not evidence of an actual driver fault on the RX.

The `--vulkan-field 8x32` path uses eight little-endian 32-bit field limbs
and the same curve/address contract; `test-vulkan` runs the full no-wallet
equivalence gate against CPU references for that selected representation.
It is intentionally opt-in until an RX self-test and matched wall profile
confirm it.

This is deliberately a simple 32-bit buffer ABI. Before performance work,
benchmark the unpack/pack cost and consider an aligned packed layout that
the curve, Keccak, SHA-256 and Base58 stages can share without conversions.
Do not change byte order based only on a single address sample. Add multiple
CPU/GPU vectors covering each input/output word and the final partial word.

## Reusable backend and remaining production gates

1. The optional four-key curve shader now has batch inversion and CPU
   equivalence checks on Mesa, but no RX measurement. On llvmpipe with 30
   one-letter test words, short 0.5-second A/B/A profiles put batch 1 at
   72–84 K/s wall and 7.3–7.9 us/key curve, versus batch 4 at 131 K/s wall
   and 2.9 us/key curve. These are software-driver numbers, not evidence of
   an RX speedup. The matched RX A/B/A now puts batch 4 at 2.985 M/s wall
   versus 2.885 M/s for batch 1, still about 35x below OpenCL. If batch 4
   loses on a future RX run, inspect VGPR/scratch and consider separate
   point/affine stages, a smaller batch, or a cooperative inversion layout.
   Verify random-base rollover and long-running searches. The normal Windows
   ZIP and opt-in CI bundle have `bench-vulkan.cmd`: a bounded no-wallet
   correctness gate followed by interleaved, repeated OpenCL/Vulkan 1/4 wall
   profiles on its adjacent dictionary. Copy the actual 358-word `words.txt`
   into that separate test folder first; do not use the starter file for an
   RX-versus-OpenCL claim.
2. `vulkan/vulkan_backend.cpp` reuses descriptors, pipelines and buffers across
   bounded batches while recording the exact dispatch count and push constants
   per submission. The default GPU batch is 131,072 keys (configurable to
   32,768/65,536/131,072/262,144/524,288/1,048,576), and the default resident
   command group records sixteen batches by default before one fence wait; the
   `--vulkan-resident-group 4|8|16` option makes that A/B tunable. This reduces fixed
   `vkQueueSubmit`/fence work without duplicating the intermediate buffers. The
   all per-key public/hash/address stage buffers, including the projective
   Jacobian scratch, are in a separate GPU-only storage buffer and are not
   mapped by the host; the profile reports whether its memory type is
   device-local. The ring records also carry the GPU-generated address, so collection does not
   read the full address buffer for every pass. The profile reports host
   setup/record/submit/fence/collection intervals so wall-rate loss can be
   separated from shader time. It resets and drains the atomic ring once per
   resident group. The OS CSPRNG base expands into a 22-bit offset
   window; every candidate scalar, address and full dictionary match is
   rechecked on CPU before output. Metadata overflow and driver timeouts stop
   the search. The `Backend` interface and explicit `--backend vulkan` CLI
   are wired; no CPU fallback occurs. Software Vulkan is rejected in normal
   runs and allowed only for tests or with an explicit developer environment
   override. Mesa tests cover repeated dispatches,
   offset boundaries and a no-wallet call through the production backend.
   The host now prefers device-local coherent host-visible memory when
   available, and `--vulkan-profile` reports wall throughput, stage GPU
   timestamps and memory locality without wallet output. On a discrete GPU
   exposing only system-memory host mappings, a staging-buffer design is a
   performance candidate; do not infer its benefit without a matched A/B.
   A Mesa llvmpipe smoke profile with 30 single-character test words
   processed 65,536 keys in 0.845 s (77.6 K/s wall); it reported
   7,526 ns/key curve and 4,454 ns/key dictionary match. This validates the
   instrumentation path, **not** RX performance: both the software driver
   and artificial word mix differ from the target workload.
   A synthetic `VK_ERROR_DEVICE_LOST` at submit is covered; varied production
   batch sizes, actual driver faults and long-running rollover tests remain.
3. Windows Vulkan SDK CI compiles the optional backend, and the no-wallet
   full-address pipeline passes on the actual RX 9070 XT. The matched resident
   runs measured 123.18 M/s for curve 1 / affine 4 / group 8, 126.09 M/s for
   the same math with group 16, 141.79 M/s for curve 4 / affine 4 / group 16,
   and 157.14 M/s for curve 4 / affine 8 / group 16. The run uses host-visible
   device-local memory and batch 131072. A 262144-key probe with group 16
   failed at `vkAllocateMemory -2`, so it is not a stable default. The current
   curve shader uses explicit libsecp256k1-style 10x26 multiply/square
   schedules and carries the 64-byte public base point in push constants. A
   long-running funded-wallet run remains unvalidated.
   The experimental 8x32 field profile uses a bounded 4096/8192/16384-key
   submit (`--vulkan-batch-keys 4096` is the CI and first-run choice); larger
   8x32 profile submits are rejected early because some Windows AMD drivers
   can hold the first dispatch behind the watchdog. This profile limit does
   not restrict production search.
Khronos's [compute guide](https://docs.vulkan.org/guide/latest/compute_shaders.html)
and [shader interface specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html)
define dispatch/descriptor requirements. If experimenting with `clspv`, use
its [OpenCL-C mapping](https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md)
and inspect the reflected bindings; `clvk` is a compatibility route, not the
native backend above. Current upstream `clspv` documents the old
`-descriptormap` flag as removed: compile the isolated `resident_stage_curve`
and `resident_stage_affine` OpenCL sources ahead of time, then run
`clspv-reflection` on each SPIR-V module. Confirm the generated binding map,
`Int8`/`Int64` and variable-pointer capabilities against the actual RX Vulkan
features before writing a host descriptor layout. This is an *experiment*, not
a claim that the complete OpenCL source compiles or runs unchanged. If it
fails or performs poorly, port the 10x26 field/group math to GLSL and keep the
same 128-byte point wire layout. Either path needs the CPU/libsecp256k1
boundary tests above before any search output is enabled.
