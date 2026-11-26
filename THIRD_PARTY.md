# Source and dependency provenance

The project license is Apache-2.0; [LICENSE](LICENSE) and [NOTICE](NOTICE) retain
upstream terms and attribution. This file records the first C port slice.

| Material | Source | Treatment |
|---|---|---|
| Retained CUDA kernels/bridge, Zig oracle and build | tensorfish/squid5090 `d10c0abb2c8990be81c08ba84c66a8333a691b33` | Unmodified, checked by `UPSTREAM.json`; Apache-2.0 and existing NOTICE chain |
| `src/safetensors.c` | Translation/rework of pinned `src/safetensors.zig` | Apache-2.0; original author and changes identified in file header |
| `src/cuda_abi.h`, `test/cuda_abi_c.c`, `test/cuda_abi_probe.cu` | Generated from `abi/cuda.json`; initial declarations derived from pinned `src/cuda.zig`, compared against actual CUDA bridge | Apache-2.0; generated attribution and exact-byte CI verification |
| `.clang-format` | artalis-io/jev-bush `85fd51f8c99b1b8b7ea26d33e02413e681a9271d` | Copied formatting policy, 0BSD |
| Makefile/API/error/testing conventions | Same Jev Bush revision | Adapted engineering patterns; no Jev model/numerical engine imported |
| `src/tokenizer.c`, `src/tokenizer_load.c` | Translation/rework of pinned `src/tokenizer.zig` | Apache-2.0; original author, changed validation/ownership and preserved lexer/merge behavior identified |
| `src/inkling.c` | Architecture/role binding derived from pinned `src/weights.zig` and `src/constants.zig` | Apache-2.0; layout validation only, no packing math replacement |
| `src/tokenizer_profile.h` | Public Inkling ByteLevel configuration | Supported-profile schema only; no vocabulary, merges or model bytes embedded |
| `src/json.c` | Extracted first-port JSON primitives plus strict configuration comparison | Apache-2.0; reused across checkpoint and tokenizer readers |
| Other C host/test/helper additions | Artalis | Apache-2.0 |
| CUDA runtime/cuBLAS/toolkit | Operator's installed NVIDIA toolkit | Not vendored; NVIDIA terms govern separately |
| Inkling model/tokenizer | External operator-supplied checkpoint | Not included in source/build contexts; see model license/AUP links in design |

The retained upstream NOTICE identifies ingine, AtomicBot and optimization-skill
influences. Retain that attribution chain; changing the host language does not
remove it. Neither model payloads nor SDK binaries are redistributed here.

Jev Bush's 0BSD grant for copied formatting/adapted build conventions:

> Copyright (C) 2026 Artalis
>
> Permission to use, copy, modify, and/or distribute this software for any
> purpose with or without fee is hereby granted.
>
> THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
> REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
> AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
> INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
> LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
> OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
> PERFORMANCE OF THIS SOFTWARE.
