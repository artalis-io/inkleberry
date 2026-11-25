# C tokenizer and checkpoint binding — validation

25 November 2026. This implements the next P1 input slice after the foundation.
The CUDA kernels, bridge and original Zig sources remain byte-identical to
`d10c0abb2c8990be81c08ba84c66a8333a691b33`. No C generation, GPU packing or
throughput result is claimed. [Machine-readable evidence](../benchmarks/input-port-2026-11-25.json).

## Implemented

- Byte-level BPE vocabulary/merge loading, special-token matching, raw-byte
  decoding, the pinned Unicode lexer and long-input chunking, six named effort
  presets for the single-user template, and reusable per-call scratch ownership.
- Bounded JSON primitives shared with the safetensors reader. Configuration
  comparison rejects unsupported normalizers, regexes and added-token flags;
  malformed IDs, missing byte tokens, duplicate IDs/names and invalid merges
  return errors rather than upstream assertions or silent fallback.
- Architecture-specific checkpoint admission/binding for 822 main text-engine
  roles and 54 optional MTP roles. Exact dtype, rank and shape are checked before
  publishing the handle. All mappings are readonly and owned by the binding;
  returned data/metadata are borrowed. Explicit MTP failures never silently fall
  back to AR-only mode. 422 vision tensors in the real checkpoint are excluded.
- CLI `tokenize TOKENIZER.json TEXT` and `validate-model MAIN [MTP]`, alongside
  the existing `inspect` command. These tools require no CUDA runtime.

## Executed tests

| Check | Result |
|---|---|
| macOS ARM64 and core-1 Linux AMD64 normal builds | Full `make check install-check` passed, including C/C++ and static/shared consumers |
| Existing checkpoint/parser regression cases after JSON extraction | 46 format cases and 3 filesystem rejection cases passed |
| Synthetic tokenizer | BPE ranks/parallel pair merging, special enabled/disabled, sizing queries, insufficient/exact buffers, invalid UTF-8, embedded NUL, copied-input ownership and decoding passed |
| Tokenizer failure paths | 35 malformed/unsupported JSON cases passed with custom allocator cleanup checked |
| Allocation faults | Every tokenizer table, scratch allocation and checkpoint binding allocation site exercised; no live owned allocations after failures |
| Allocation-free execution | 100 repeated encode/decode checks after warmup with the allocator set to fail on the next call; no allocation attempted |
| Concurrency | Shared immutable tokenizer with two distinct scratch handles passed; Linux ThreadSanitizer also passed |
| Model schema | Every one of 876 descriptors accepted its expected shape and rejected rank/dtype/dimension changes |
| Real-file binding regressions | Synthetic sparse files passed; 10 layout/role drift cases, missing MTP, reversed header order and ignored vision extras behaved correctly |
| Linux ASan/UBSan/leak checks | Full suite passed |
| Linux libFuzzer | 3,000 runs each for safetensors and tokenizer load/encode/decode, under ASan/UBSan, passed |
| Formatting/invariants/provenance | Passed with both clang-format 18.1.8 locally and Ubuntu 18.1.3 extracted into an isolated tools directory |
| Real tokenizer | 14/14 cases matched both pinned Zig and provider `tokenizers`; all decoded bytes round-tripped |
| Real template | All six effort templates produced the same token IDs as Zig |
| Real checkpoint binding | All 876 bound descriptors/data spans matched independent Python header inspection; all 422 vision descriptors remained unbound |
| Real checkpoint allocation faults | The same five-site binding fault suite passed against the external main and MTP files |

The real tokenizer corpus includes empty/ordinary English, Hungarian, Japanese,
Arabic, emoji/combining characters, code, contractions, digits, whitespace,
special tokens and embedded NUL. A 70,000-byte single piece exercises the BPE
piece boundary; 1,215,000-byte and 1,260,000-byte texts exercise chunking. These
are real input strings, not synthetic KV allocations or inference-context tests.
The two large texts produced 360,000 and 360,001 tokens respectively.

The model and tokenizer stayed external at
`/data/models/safetensors/inkling-2-mini-c9d028302cd0`. Core work is isolated under
`/data/scratch/squid-c-port-20261125`. Provider comparison used an existing Python
interpreter read-only with bytecode writes disabled; no packages, drivers,
services, power limits or model files were modified. No GPU was used for this
slice's input validation. Model bytes/vocabulary/merge tables are not included in
this repository. Sparse fixtures contain zero-filled holes and a synthetic
metadata header; a small probe checks sparse-file support before creating them.

## Reproduction

See README for host and external input tools. Run sanitizer targets sequentially
because each cleans `.build`:

```sh
make -j8 check install-check
make lint
ASAN_OPTIONS=detect_leaks=1 make sanitize
ASAN_OPTIONS=detect_leaks=1 make fuzz-smoke
make thread-sanitize
python3 tools/check_inkling.py ./libsq.so /external/model
python3 tools/check_tokenizer.py ./libsq.so /external/model \
  --zig-oracle /external/pinned/squid5090 --reference
```

Rebuild a normal shared library (`make -j8`) before loading it under ordinary
Python after sanitizer runs. Core-1's incomplete auto-selected GCC 14 runtime
needs `SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'` for
libFuzzer; no global installation is needed. MacOS sanitizer runtime startup
remains unavailable locally as documented in the first-slice report; no new
local macOS sanitizer pass is claimed.

## Limits and remaining gates

The lexer is a faithful translation of Zig's category ranges, not a complete
Unicode regex engine. The representative corpus does not establish exhaustive
Unicode/provider parity. `ignore_merges` follows the pinned Zig merge graph.
Arbitrary tokenizer normalization, numeric effort strings, general multi-turn
and tool templates are not supported yet. Scratch is exclusive per active call;
tokenizer/binding close requires exclusive lifetime ownership.

Binding establishes model-role/layout compatibility, not full numeric payload
validity, checkpoint authenticity, config.json compatibility or tokenizer/model
identity. These input APIs are separate; a future combined model loader must
validate their association and generation token ranges. GPU upload/packing,
auxiliary ablation/pruned-vocabulary handling, numerical packing parity and
forward/decode remain subsequent work. P1 is not fully closed until real weight
packing parity has passed; P2 generation has not started.
