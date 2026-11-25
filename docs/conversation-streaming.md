# Text conversations and streaming foundation

The host library now exposes `sq_chat.h`. This slice precedes HTTP: transport,
worker admission, authentication and SSE framing remain separate. The HTTP
adapter is to use the latest Keel release, resolved and pinned when implemented.
Original CUDA and Zig sources remain at the hash-verified evaluated revision.

## Conversation encoding

`sq_chat_encode` accepts counted typed text messages: system, developer, user and
assistant, with optional names and historical reasoning. Initialize options with
`SQ_CHAT_OPTIONS_INIT`. Defaults select thinking effort none, omit historical
reasoning and append Inkling's model/text generation prefix. `keep_history_thinking`
and `enable_thinking` are explicit booleans. All six existing thinking-effort
presets are supported.

The encoder matches the pinned server's text-message layout, including leading
system/developer messages, effort placement, assistant reasoning/content blocks,
model-end markers and the system-prefix token count. That count can be passed to
`sq_decode_options.system_prefix_tokens` for the existing opt-in prefix cache.
Caller text and names use ordinary tokenization; only library-controlled protocol
markers enable special-token parsing. Literal protocol-looking user text cannot
insert control IDs. Ordinary vocabulary and special-marker lookup remain separate.

The caller owns immutable inputs, exclusive tokenizer scratch and output. No
allocation occurs during encoding. There must be 1–1,024 messages and at most
32 MiB of combined content/reasoning/name bytes; names are limited to 128 bytes.
UTF-8 is strict. Unsupported role values, missing required content, invalid flags,
reserved fields and over-limit inputs fail explicitly. The complete token count
and system-prefix count are returned by sizing/short-output calls; output stays
untouched when capacity is insufficient. No trailing messages are dropped.

A NULL assistant content pointer denotes absent content, while a non-NULL empty
span denotes empty text. An empty assistant text block is omitted when a kept,
nonempty reasoning block exists, matching Zig. Reasoning is intentionally omitted
unless the keep-history option is enabled. Tools, images and content-part arrays
are outside this typed text interface; a future JSON adapter must reject fields
it cannot represent. The legacy single-user renderer/CLI behavior is unchanged.

## Shared streaming and collected-output filter

`sq_stream_create` owns a bounded filter through the injected allocator and
borrows an immutable tokenizer until destruction. Byte-only callers can use a
NULL tokenizer with `parse_chat=0`. The object copies up to eight stop strings,
each 1–64 valid UTF-8 bytes. Pushes allocate nothing and are limited to 32 MiB per
byte call; the filter does not accumulate the whole answer.

Token input recognizes actual Inkling control IDs, suppresses the model header,
separates reasoning/content callbacks and recognizes EOS. Unsupported generated
tool, multimodal or unknown control sections fail explicitly. Header bytes have
a 1,024-byte bound. Byte input uses the same Unicode/stop path for raw completion
text. Do not mix byte and token input in one request.

The incremental decoder emits complete UTF-8 code points. Default replacement
uses Unicode maximal-subpart behavior, checked against Python's
`decode(errors='replace')`; strict mode reports FORMAT. A partial character at
finish or a protocol section boundary follows the same configured policy.
Content stop matching spans calls and content sections, including cases where
an earlier incomplete stop prefix overlaps a later completed stop. No stop bytes
are emitted. Reasoning receives Unicode decoding but no content stop matching.
One filter is used for both streaming callbacks and collected output.

Callbacks borrow complete UTF-8 chunks until they return. Consume the chunk
before returning zero to stop. A stream has one owning thread; reentry from its
callback returns BUSY. Decoder errors remain sticky until reset, while invalid
arguments leave state unchanged. Reset/destroy discard pending bytes without
calling the callback; finish flushes unmatched prefixes and final Unicode state.
Reset reuses the configured stops and allocation. Finish is idempotent.

Initialize diagnostics with `SQ_STREAM_INFO_INIT`. Counts describe consumed
model token IDs (including control/EOS), produced content/reasoning bytes and
Unicode replacements. They are not network-delivery receipts. Finish reasons
separate stop, EOS, length, callback, cancellation and errors. An engine callback
can stop when the filter finishes; a future transport adapter must use filter
finish reasons when mapping EOS/custom stops, since the engine may report its
callback stop. Client disconnects must separately request engine cancellation.

## Verification and reproduction

```sh
make -j8 check install-check
make lint                         # clang-format 18
make sanitize
make thread-sanitize
make fuzz-smoke                    # adds a stream partition fuzzer
make ZIG=/path/to/zig chat-oracle
python3 tools/validate_chat.py --model /external/model-directory \
  --library ./libsq.so --oracle .build/chat-oracle
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 .build/chat-native
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  .build/chat-native /external/model-directory
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  .build/chat-native /external/model-directory
```

Use `libsq.dylib` on macOS. The oracle needs the already available Zig 0.17.1;
the C runtime does not. The test adapter copies pinned Zig sources under
`.build` and changes visibility only, exposing the actual renderer and buffers.
It does not replace the renderer with a Python reconstruction or patch originals.

The real tokenizer matched Zig in 108 cases: six conversation fixtures, six
efforts and three reasoning/generation configurations. These include Unicode,
NUL, protocol-looking text, system/developer placement, absent/empty assistant
content, names and the 1,024-message boundary. Ninety Unicode/stop fixtures match
an independent replacement/leftmost-stop reference across chunk partitions.
CPU tests cover errors, limits, output sentinels, long tokens, channel boundaries,
stop ownership, callback BUSY, OOM and reset. The stream fuzzer checks one-shot
versus byte-by-byte output under ASan/UBSan, alongside the existing parsers.

Native Inkling on the reserved RTX PRO 6000 generated `7` for a multi-turn
arithmetic conversation. Streaming/collected output and cold/warm generated IDs
matched; 846 prompt tokens were reused. Custom stop and a subsequent full replay
passed. Full native memcheck reported zero errors and zero leaked bytes.
See [the evidence manifest](../benchmarks/chat-2026-11-25.json) for exact scope.
These checks establish this text interface, not broad OpenAI/tool compatibility,
a performance benchmark or qualification of an HTTP service.
