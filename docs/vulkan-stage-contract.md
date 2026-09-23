# Native Vulkan pipeline: stage contract and verification gates

The native Vulkan wallet backend is **experimental**. Source builds with
`-DTRON_ENABLE_VULKAN=ON` have a generic dispatch probe, a native secp256k1
10x26 field-math point stage, Keccak-256, SHA-256d, Base58Check and flattened
dictionary matching. The curve stage can compute `P + G` or walk from one
base point through three 8-bit offset-table windows; it currently inverts
each result individually, not with the faster OpenCL batch inversion.
The generic and deterministic stage tests do not save wallets. A separate
reusable `--backend vulkan` path now runs full-address searches or no-wallet
benchmarks, but is not validated on the RX 9070 XT. The regular Windows
release remains OpenCL/CUDA only.

## Verified interface to preserve

`vulkan/curve.comp` reads 16 little-endian words per affine base point at
binding 6 and a `3 × 256 × 16`-word table at binding 7. Mode 1 computes
`P0 + offset·G`, where the offset is a push-constant base plus the invocation
ID, and writes 16 words per public key at binding 0. Mode 0 tests `P + G`,
including doubling at `P = G`. The host derives every expected point
independently with `libsecp256k1`; tested offsets cross 255/256, 65535/65536
and end at `2^22 - 1`. A compute barrier separates curve from Keccak.

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
`vulkan/match.comp` reads text at binding 3 and packed DFA/output tables at
binding 4. In test mode it writes one 18-word record per address at binding 5
(count, overflow, 16 distinct word IDs). In ring mode it atomically reserves
a slot in metadata binding 8 and writes a 20-word record at binding 9 (key
index, count, flags, reserved, 16 IDs). Tests use capacity 8 plus two canary
guard slots, compare concurrent records without assuming write order, cover
exact-capacity and over-capacity cases, and check both overflow flags. The
host constructs the 30 embedded one-letter test words through the production
Dictionary builder (the same list is in `tests/vulkan_words.txt`)
and compares records with `Dictionary::matchIds()`. This is still a *test*
ring: no scalar/address is emitted to user output. Capacity overflow is
fail-closed; a truncated ID list can be recovered only if the production host
recomputes every match from the full CPU dictionary.

This is deliberately a simple 32-bit buffer ABI. Before performance work,
benchmark the unpack/pack cost and consider an aligned packed layout that
the curve, Keccak, SHA-256 and Base58 stages can share without conversions.
Do not change byte order based only on a single address sample. Add multiple
CPU/GPU vectors covering each input/output word and the final partial word.

## Reusable backend and remaining production gates

1. Move the verified single-point curve math to a staged point/affine layout
   with batched inversion. Compare every scalar/public key with CPU
   `libsecp256k1`, including random-base rollover; measure actual full-wall
   speed before selecting a batch size. The current `P0 + offset·G` shader is
   correct on Mesa but not yet a competitive resident pipeline.
2. `vulkan/vulkan_backend.cpp` now reuses descriptors, pipelines, buffers and
   command objects across bounded batches. It resets and drains the atomic
   ring every dispatch. The OS CSPRNG base expands into a 22-bit offset
   window; every candidate scalar, address and full dictionary match is
   rechecked on CPU before output. Metadata overflow and driver timeouts stop
   the search. The `Backend` interface and explicit `--backend vulkan` CLI
   are wired; no CPU fallback occurs. Software Vulkan is rejected in normal
   runs and allowed only for tests or with an explicit developer environment
   override. Mesa tests cover repeated dispatches,
   offset boundaries and a no-wallet call through the production backend.
   Driver-error injection, varied production batch sizes and long-running
   rollover tests remain to be added.
3. Windows Vulkan SDK CI now compiles the optional backend, but runtime has
   only been verified with Mesa software Vulkan on Linux. Test the stage
   executable on the actual RX 9070 XT before enabling any production use.
   Once the full backend exists, compare full wall keys/s against
   the same dictionary and GPU workload under staged OpenCL; also measure
   stage ns/key and transfer overhead. Do not recommend Vulkan on performance
   grounds until correctness and matched A/B/A runs pass.

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
