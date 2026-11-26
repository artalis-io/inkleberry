# Reference material

This tree is separate from the maintained C library and excluded from production
source, embedding and runtime packages. Nothing here is started by normal builds.

| Directory | Purpose |
| --- | --- |
| `upstream/` | Frozen implementation/build script and original documentation |
| `oracle/` | Optional comparison harnesses |
| `validation/` | Scripts driving those comparisons from the repository root |
| `tools/` | Inherited experimental Python utilities and their optional environment |
| `design/` | Historical architecture/port designs and superseded interface draft |
| `reports/`, `benchmarks/` | Qualification reports and immutable result receipts |
| `docs/`, `assets/` | Historical upstream plans, notes and illustrations |
| `deploy/` | Legacy service example for the reference implementation |

`UPSTREAM.json` at the repository root preserves the exact revision, original
file names and hashes. Its relocation map identifies the frozen files here.
`upstream/cuda` points to the single canonical `cuda/` directory; kernels are not
duplicated. LICENSE, NOTICE and THIRD_PARTY.md at the root retain attribution.

## Optional comparisons

Run from the repository root with an explicitly installed reference compiler:

```sh
make reference-check
make ZIG=/path/to/zig ar-oracle chat-oracle
python3 reference/validation/validate_ar.py --model /external/model-directory \
  --cuda-lib /usr/local/cuda/lib64
```

Weight, decode, adaptive and prefix oracle targets are also defined in
`oracles.mk`. Actual GPU runs require the approved idle-device UUID through the
normal guard; do not run comparisons on a busy device. These tests remain useful
coverage and are not a second maintained production engine.

To build the frozen executable itself, enter `upstream/` and run `/path/to/zig
build`. The legacy deployment example is not a service definition for the C
library. No reference service should be started by production packaging.

The experimental utilities in `tools/` retain their original behavior/attribution.
Their third-party environment is optional and isolated in that directory; it is
not required for library development or runtime. Experiments and old reports
refer to their recorded revision, commands and benchmark inputs. Do not rewrite
historical JSON receipts as if they described a new build.

The maintained product is now **Inkleberry Finn**, with the `ifn_` public API.
Historical names below this directory remain archival; optional validation
adapters and `oracles.mk` target the current build outputs.

Native comparison targets prepare disposable Zig copies with adapted C link names.
The original Zig sources and their hashes remain unchanged. Current C/CUDA
sources, including kernels and bridge types, use the project namespace throughout.
