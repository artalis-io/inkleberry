# Inkleberry Finn naming

Repository: https://github.com/artalis-io/inkleberry. The local checkout directory
can be named freely; examples use `inkleberry`.

| Surface | Name |
| --- | --- |
| Project | Inkleberry Finn |
| Public functions/types | `ifn_*` |
| Public constants/macros | `IFN_*` |
| Headers | `ifn.h`, `ifn_chat.h`, `ifn_engine.h`, `ifn_gpu.h`, `ifn_weights.h`, `ifn_infer.h`, `ifn_infer_inkling.h` |
| Embedding distribution | `dist/ifn.h` + `dist/ifn.c` |
| Host library | `libinkleberry.a`, `libinkleberry.so` / `.dylib` |
| Inference library/provider | `libinkleberry-infer.a`, `libinkleberry_backend.so.1` |
| Native archives | `libinkleberry-gpu.a`, `libinkleberry-cuda.a` |
| CLI | `inkleberry`, `inkleberry-generate`, `inkleberry-weights` |
| pkg-config | `inkleberry`, `inkleberry-infer` |
| Backend identity | `artalis.inkleberry` |
| Backend extension | `ifn_infer_inkling_request_v1`, `IFN_INFER_EXT_INKLING_REQUEST` |
| Test environment | `INKLEBERRY_TEST_GPU_UUID`, `INKLEBERRY_TEST_TIMEOUT_SECONDS` |
| Release archives | `inkleberry-source-*`, `inkleberry-embedding-*`, `inkleberry-runtime-*` |

This is a deliberate breaking pre-release rename: no aliases for the former
`sq_*` API, old headers, backend identifier, library names or CLI are provided.
Rebuild consumers against the new headers/libraries; do not mix old objects with
new libraries. ABI structure/version semantics otherwise remain unchanged.

All project-owned native C/CUDA functions, internal helpers, namespaces, types,
constants and macros use `ifn_` / `IFN_`; the native environment flag is now
`INKLEBERRY_MTP_BF16`. Local variables/members, standard entry points (`main`,
`LLVMFuzzerTestOneInput`) and identifiers supplied by CUDA/the standard library
keep their external conventions.

The retired Zig implementation and comparison harnesses are no longer included.
There are no legacy C/CUDA compatibility aliases.

The native namespace map records old names as provenance data, not production
symbols. Reversing the identifier map and reviewed host-error adapters (and removing
each file's modification notice) must reproduce every original upstream CUDA hash. Legal notices, original
hashes, upstream URLs and historical documentation retain attribution.

Host qualification and numerical GPU qualification are distinct. See
[distribution validation](distribution-validation.md) for the executed model-backed modular and amalgamated qualification.


## Canonical branch and history

`main` is the integration and release branch of the standalone Inkleberry Finn
repository. The original fork and its pull-request history are retained in the
private `artalis-io/inkleberry-fork-archive` repository. Current CUDA provenance
remains in `UPSTREAM.json`, `THIRD_PARTY.md` and `NOTICE`. Upstream changes are
reviewed and qualified explicitly; there is no automatic synchronization branch.

Removing the fork relationship and retiring the original host implementation do
not change the retained license or attribution chain. Historical qualification
reports in `reference/` describe their original revisions and are not executable
validation instructions for the current tree.
