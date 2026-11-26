# C source promotion and reference layout

The C host is now the production implementation directly under `src/`. This
change relocates files and updates paths; it does not change generation behavior,
public interfaces, CUDA mathematics or the pinned upstream revision.

```text
include/                         public C headers
src/*.c, *.h, infer.exports       production C implementation and private headers
src/cuda/                        checked CUDA adapters
cuda/                            canonical unchanged upstream kernels/bridge
reference/upstream/src/*.zig      unchanged optional Zig oracle
reference/upstream/build.zig      unchanged original build script
reference/upstream/cuda           symlink to ../../cuda
```

All 40 files formerly in `src/c/` moved byte-for-byte. Only the two checked CUDA
adapter include paths changed, from `../c/checked_*.h` to `../checked_*.h`.
The original Zig build script and ten source files also moved without content
changes. No kernel copy was introduced and no license/NOTICE content removed.

`UPSTREAM.json` retains the original upstream names, exact hashes and revision.
Its `relocated_files` map identifies the new local paths. `tools/upstream.py`
resolves these for full reference verification and chat-oracle preparation. Native
ABI generation now uses [the independent schema](../abi/README.md).
Generated ABI declarations and layout probes remain byte-identical. Historical
receipts keep their recorded paths; use their original commit or this relocation
mapping when interpreting them. Do not rewrite past evidence to current paths.

After updating an existing checkout, perform one clean build:

```sh
make clean
make -j8 check install-check
make lint                       # clang-format 18 required
```

Old `.build/*.d` files refer to the previous paths. An initial incremental build
against those files failed with the expected missing-source dependency; the clean
build passed. No compatibility shim or old source copies are retained.

Normal production builds and checks need neither a Zig compiler nor Zig source
files. `make generated-check` verifies production pins and the independent ABI
schema; `make reference-check` additionally verifies the optional pinned reference.
Zig itself is needed only for explicit oracle targets.
See [reference operations](../reference/README.md) for those commands and for
building the original upstream executable. The legacy ablation tool's default
reference executable/corpus paths were updated; its algorithm was not changed.

## Executed validation

Compared against pre-cleanup commit `f2810074f0aaa3529987f1fd02eb6f233d40f9e1`:

- Byte equality for every promoted file; all original upstream hash entries and
  revision unchanged; one canonical CUDA directory verified.
- Clean macOS ARM64 and Linux AMD64 host/installed-consumer checks; formatting,
  provenance and generated declarations; changed Python tooling syntax checks.
- Linux ASan/UBSan, ThreadSanitizer and four fuzz targets, 3000 runs per target.
- Native CUDA C/C++ ABI probe: 76 function prototypes, seven records, 72 layout checks.
- Real Inkling shared-provider workflow on the approved idle RTX PRO 6000:
  adaptive/lookup generation, independent request streams, stale/active
  cancellation, cache reuse and scope isolation, stop withholding and replay.
  Answer `7`, 876 prompt tokens, three generated IDs, 846 tokens reused.
- Relocated C/Zig AR oracle: NVFP4, FP8 KV, graphs, 33 real prompt tokens;
  1,000,290 logits compared and 32 generated tokens; caller context preserved.
- All 108 real-tokenizer chat-template comparisons against the relocated oracle.
- Original reference `zig build` succeeded with the canonical CUDA symlink.
- Optional weight, AR, decode, adaptive, prefix and chat oracle targets compiled.
- Shared-provider export allowlist, SONAME and ctypes discovery passed.

Native builds used isolated `/data/scratch/squid-c-layout*-20261126` directories;
models remained external. The GPU returned to its original 2 MiB usage, and the
other GPU stayed at 26 MiB. No services, drivers or system configuration changed.

Sanitizer builds used the installed GCC 13 runtime search path via
`SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'` and the matching
`SAN_CXX`, as in the previous ABI qualification. The full long-context/precision
benchmark matrix, GPU memcheck and legacy ablation workflow were not repeated for
this path-only change. Existing prior qualification evidence is preserved.

## Retiring Zig later

Keep this reference until the remaining Zig parity checks have independent,
maintainable replacements. The ABI generator's former dependency on pinned Zig
declarations has been removed using the language-neutral schema. The optional
oracle coverage remains operational until its independent replacements are ready.
