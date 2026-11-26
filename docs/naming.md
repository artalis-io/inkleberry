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

The pinned CUDA bridge keeps its internal upstream symbols and environment knobs
(including `SQUID_MTP_BF16`) to preserve its verified bytes. These are not the
public inference ABI. Legal attribution, `UPSTREAM.json`, archived reports and
frozen reference sources preserve their historical names. Optional reference
validation adapters use the current executable names.

Host qualification and numerical GPU qualification are distinct. See
[distribution validation](distribution-validation.md) for the executed model-backed modular and amalgamated qualification.
