# Native Vulkan pipeline: stage contract and verification gates

The native Vulkan wallet backend is **experimental**. Source builds with
`-DTRON_ENABLE_VULKAN=ON` have a generic dispatch probe, a native secp256k1
10x26 field-math point stage, Keccak-256, SHA-256d, Base58Check and flattened
dictionary matching. The curve stage can compute `P + G` or walk from one
base point through three 8-bit offset-table windows; it currently inverts
each result individually by default. An optional four-key shader shares one
Montgomery inversion across four Jacobian Z values. It is built as separate
SPIR-V so its larger live arrays cannot inflate register pressure in the
default one-key variant. Both variants have now been profiled on the RX 9070 XT
with the same 358-word dictionary; correctness passed, but wall throughput was
only about 2.9-3.0 M keys/s versus about 104.6 M/s for staged OpenCL.
The generic and deterministic stage tests do not save wallets. A separate
reusable `--backend vulkan` path now runs full-address searches or no-wallet
benchmarks, and has passed a no-wallet correctness/profile run on the RX
9070 XT. The regular Windows
release now includes the Vulkan code, while its adjacent config still selects
OpenCL by default. Vulkan requires an explicit backend selection.

## Verified interface to preserve

`vulkan/curve.comp` reads 16 little-endian words per affine base point at
binding 6 and a `3 × 256 × 16`-word table at binding 7. Mode 1 computes
`P0 + offset·G`, where the offset is a push-constant base plus the invocation
ID, and writes 16 words per public key at binding 0. Mode 0 tests `P + G`,
including doubling at `P = G`. The host derives every expected point
independently with `libsecp256k1`; tested offsets cross 255/256, 65535/65536
and end at `2^22 - 1`. A compute barrier separates curve from Keccak.
Mode 2 maps each invocation to up to four consecutive offsets. It now builds
the first point from the offset table and walks `+G` for the remaining three
points, avoiding redundant table lookups and mixed additions. It stores the
four Jacobian points and prefix products of their Z coordinates, inverts the
final product once, then walks backward to derive each `1/Z`. Partial groups
of 1, 3 and 5 keys, table-window boundaries and the last valid offset are
checked against CPU addresses. The host dispatches only `ceil(count/4)`
invocations for the curve stage; later address stages still dispatch `count`.

All production stages read the active key count from runtime storage binding
10 (`words[0]`); the curve stage reads the current offset base from
`words[1]`. These values are updated between submissions, so the command
buffer and push-constant state remain static. `vulkan/keccak.comp` takes 16
little-endian `uint32_t` words per 64-byte uncompressed public key (`X||Y`,
no `0x04` prefix) at storage binding 0. It
outputs six little-endian words at binding 1: byte `0x41`, the last 20 bytes
of Keccak-256, and three zero padding bytes. The first push-constant word is
reserved for compatibility; the active count comes from binding 10. The test
uses up to 257 deterministic, test-only secp256k1 scalars. It dispatches counts 1, 8, 63, 64, 65 and 257, comparing every active
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
`vulkan/match.comp` reads text at binding 3 and packed DFA/output tables at
binding 4. In test mode it writes one 18-word record per address at binding 5
(count, overflow, 16 distinct word IDs). In ring mode it atomically reserves
a slot in metadata binding 8 and writes a 20-word record at binding 9 (key
index, count, flags, reserved, 16 IDs). Tests use capacity 8 plus two canary
guard slots, compare concurrent records without assuming write order, cover
exact-capacity and over-capacity cases, and check both overflow flags. The
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
2. `vulkan/vulkan_backend.cpp` now reuses descriptors, pipelines, buffers and
   one pre-recorded command buffer across bounded batches. The per-batch count
   and offset are 16 bytes in a small runtime-parameter storage binding, so the
   CPU no longer resets, begins, records or ends a command buffer for every
   dispatch. The default submit batch is 131,072
   keys (configurable to 32,768/65,536/131,072/262,144); this reduces fixed
   `vkQueueSubmit`/fence work without changing the GPU math or result ABI. The
   profile also reports host setup/record/submit/fence/collection intervals so
   wall-rate loss can be separated from shader time. It resets and drains the atomic
   ring every dispatch. The OS CSPRNG base expands into a 22-bit offset
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
3. Windows Vulkan SDK CI compiles the optional backend, and a no-wallet
   full-address profile now also passes on the actual RX 9070 XT. The profile
   reports host-visible device-local memory; GPU stages reached 24.73 M/s for
   batch 4 on average, but roughly 4.4 s of each five-second run remained
   host/queue/transfer overhead. Do not recommend Vulkan on performance
   grounds until the larger-submit A/B is measured: the next target is
   reducing per-dispatch synchronization and host work, not changing the
   curve batch or staging memory blindly. `bench-vulkan.ps1 -VulkanBatchKeys
   32768` reproduces the old baseline; `-VulkanBatchKeys 131072` exercises the
   larger submit. A long-running funded-wallet run remains unvalidated.

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
