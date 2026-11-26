# Inference provider ABI

The experimental v1 facade is implemented in `include/sq_infer.h`. Native GPU
execution remains Linux AMD64/Blackwell; host tests run without CUDA. The original
parser, tokenizer, generation CLI and low-level engine APIs remain available.
CUDA mathematics and the pinned upstream files are unchanged.

## Multiple streams

Each request owns its tokenized input, output filter, sequence numbers,
cancellation token and terminal result. Several requests may be prepared at once;
each run supplies its own callback/context. Completing, cancelling or closing one
request cannot redirect output or cancellation to another. Requests are one-shot.

**Independent streams do not imply simultaneous GPU execution.** Discover
`max_prepared_sessions`, `max_active_runs` and `limit_scope`. The current provider
allows one prepared native session and one active run per process. A competing run
returns `SQ_BUSY` and leaves that request prepared. The host queues it and retries
later; the backend does not spawn threads or implement an admission queue.
Preparing/closing a request also returns BUSY while its session executes. Prepare
queued work between runs; a server can retain unprepared input in its own bounded
queue while a run is active.

This contract permits a future provider to advertise more execution capacity
without changing callbacks or request ownership. It does not promise CUDA-stream
parallelism, continuous batching, multi-GPU execution or fairness today. A server
must bound queues, impose deadlines and handle disconnects. Keep one provider
instance loaded per process, for its lifetime. Session limits are not permissions
to load duplicate providers and bypass native resource constraints.

## Building and consuming

```sh
make check install-check                   # CPU tests, mock execution, C/C++/FFI
make infer                                 # libsq-infer.a; no CUDA needed to build
make gpu                                   # existing native backend, needs CUDA
make infer-shared                          # Linux libsq_backend.so.1, CUDA required
make infer-provider-check                  # exported ABI, SONAME and real-provider FFI
make DESTDIR=/your/stage PREFIX=/usr/local install-infer-backend
```

The dynamic provider exports only `sq_infer_get_api`, versioned `SQ_INFER_1.0`,
with SONAME `libsq_backend.so.1`. Internal and CUDA symbols are not part of the ABI.
`pkg-config squid-infer` describes the installed shared provider. Never pass handles between different providers/function tables. Static consumers
link `libsq-infer.a libsq-gpu.a libsq.a` and the existing CUDA/C++ runtime libraries.
The CPU-only `libsq` has not acquired a CUDA dependency. Installed facade headers
and the static facade archive are also available through the ordinary install.

Negotiate before accessing the immutable function table:

```c
const void *raw = NULL;
sq_status status = sq_infer_get_api(1, 0, &raw);
if (status != SQ_OK) { /* incompatible provider */ }
const sq_infer_api_v1 *api = (const sq_infer_api_v1 *)raw;
```

Check negotiation success and table extent before calling fields. Never cast a
`T **` to `void **`. The [public-only C consumer](../test/infer_consumer.c),
[ctypes consumer](../test/infer_ffi.py) and [native workflow](../test/infer_native.c)
exercise the same contract. No HTTP library or harness-specific types cross it.

Initialize records with `SQ_INFER_INIT`, then call `defaults` for configuration
records. Input records validate size, flags and reserved fields. Unknown zero
tails are accepted; nonzero unknown input tails are unsupported. Arrays specify
count and stride; extensible elements validate their own extents and zero tails.
Output records preserve caller capacity, report known bytes written and leave
unknown tails untouched. All counts/spans are bounded and checked. The native ABI
is not a serialized cross-platform format. Callers must supply valid accessible
buffers of the declared extents and must not overlap inputs/outputs.

## Lifecycle

1. `model_create`: host bookkeeping, copied configuration and allocator setup.
2. `model_plan`: CPU weight allocation plan; does not download or load artifacts.
3. `model_load`: local tokenizer/checkpoint validation and native weight upload.
4. `session_create`, `session_plan`, `session_prepare`: separate host bookkeeping,
   CPU arena planning and expensive native resource acquisition.
5. `request_create`: validates, tokenizes and owns all necessary request state.
6. `request_run`: synchronous execution on the caller's worker, bounded callbacks.
7. `request_result`, then `request_close`; reuse the session with a new request.
8. Close session, then model. Parent closure with live children returns BUSY.

Constructors clear output handles on failure. Load/prepare failures retain their
already-created handle for cleanup. Failed weight uploads retain checkpoint mappings until native cleanup succeeds.
A failed close retains the facade handle in
CLOSING and refuses new work; retry close. Successful close clears the pointer.
NULL handles can be closed. A fatal native run failure poisons the session. Only
close is permitted for execution resources after poisoning.

Allocator callbacks are paired, maximally aligned and valid until every child is
closed. Their context must support concurrent calls if the host uses concurrency.
Owned-host budgets cover allocations made through the facade's nested allocator,
including its bookkeeping/headers. Session budget accounting is also covered by
the parent model budget. Caller buffers, mapped checkpoint pages, CUDA-pinned host scratch, vendor internal
allocations and process RSS are not governed by these counters. These are allocator
budgets, not a hard bound on all host memory. Unknown host/vendor totals are flagged. Owned-device
budgets are checked against CPU layout plans before acquisition: model budget
covers weight-loading peak, session budget covers its execution arena. They are
separate limits, not an aggregate process/device quota. Unknown vendor
allocations are explicitly marked; plans do not reserve GPU memory or prevent
another process from consuming it. OOM may still occur during acquisition.

## Configuration and capabilities

Backend ID: `artalis.squid`. Current defaults:

- Device ID: decimal CUDA ordinal `0` in the caller's visible-device namespace.
- Weight profile: `nvfp4-ar`. Also `exact-ar`, `fp8-ar`, `exact-mtp`, `fp8-mtp`,
  `nvfp4-mtp`. “Exact” names the existing mixed representation, not universally
  full-precision weights. Model MTP profiles load DSpark weights explicitly.
- Source format: `inkling-safetensors`, a local directory containing
  `tokenizer.json`, `model-00001-of-00001.safetensors` and, for MTP, `mtp.safetensors`.
- Execution profile: `fp8-graph`. Alternatives: `bf16-eager`, `fp8-eager`,
  `q4-eager`, `bf16-graph`, `q4-graph`. Context: 8192 (valid range 64..1048576).
  The upper configuration limit does not claim qualification at every length.
- Request: greedy AR, seed zero, top-p 0.95, 256 output tokens, text output,
  4096-byte maximum events, no cache reuse and no reasoning-history replay.

Compiled capabilities, configured model support and enabled state are distinct.
Speculation is unsupported without MTP weights. Required unknown features fail.
Model/tokenizer IDs are descriptive; artifact fingerprints remain empty and
`fingerprint_verified` false until content hashing is implemented. Do not use IDs
as integrity evidence. Source files must stay immutable while loading; no remote
code or automatic downloads are executed. Profile enumeration is not implemented;
only these documented identifiers are accepted.

`sq_infer_squid.h` defines the versioned backend request extension (ID 1, version 1):
draft width 0..7, adaptive/lookup mode, confidence threshold, thinking effort and
thinking enablement. Adaptive mode requires seven drafts, graphs and a valid
threshold. Prompt lookup requires adaptive mode. Template controls require chat
input. Unknown/duplicate extensions fail; they are not ignored or JSON passthrough.
The existing precision/drafting behavior is preserved. No arbitrary fallback.

## Output, cancellation and threading

Typed text chat preserves system/developer/user/assistant roles, names, absent
versus empty content and explicit reasoning-history policy. Literal raw text uses
no chat template or special-token parsing. Raw token input validates vocabulary
IDs and requires raw-token or discard output. Tools/multimodal input are absent
from this v1 text contract and must be rejected by a transport adapter.

Text output contains committed complete UTF-8, with invalid sequences replaced
using the existing decoder policy. Content/reasoning are separate event kinds;
reasoning delivery is opt-in. Text chunks obey `max_event_bytes` (4..65536).
Stops are up to eight nonempty UTF-8 strings, at most 64 bytes each; they are copied
and matched across token/chunk boundaries. String stops with raw/discard output
are unsupported. Raw-token events include model control/EOS IDs; no draft IDs.
Discard output explicitly permits a NULL callback. Callbacks receive borrowed
spans valid only during that call. Copy them into a bounded host queue if needed.

Callback actions are CONTINUE, STOP (normal completion) and CANCEL. Unknown actions
fail. Callbacks must be prompt, must preserve the device/context, and must not
throw or longjmp through native frames. Only progress, result queries and
cancellation may be used reentrantly on the active request; result is NOT_READY
until terminal. Other operations on its session return BUSY. The host must
coordinate handle lifetime: no concurrent close versus arbitrary API entry.

`request_cancel` is thread-safe and nonblocking while the handle is alive. It
returns ACCEPTED, ALREADY_REQUESTED or TOO_LATE. It sets only that request's token;
it never forwards stale cancellation to session-global cancellation. The native
engine polls the borrowed token at existing safe points. A prepared cancellation
runs to CANCELLED without calling native generation. Completion and cancellation
race at a single atomic seal. An accepted cancellation cannot later become normal
success. GPU/reset failure can instead produce FAILED. Hold the execution lease
through drain, late cache invalidation and terminal publication. Cancellation does
not retract output and does not preempt an in-flight CUDA kernel or blocked callback.

Terminal results normalize EOS, output limit, stop string, callback stop,
cancellation and failure. Generated-token counts include controls/stop-trigger
IDs passed to the output pipeline, not speculative proposals. Content/reasoning
byte counts describe callback emission, not acknowledged network delivery. Timing
validity bits distinguish missing measurements. First-content time starts at run
entry and requires real content. Prefill timing is currently unavailable; progress
reports a conservative prompt-processed count (zero until native return), not a
continuous prefill counter. Do not expose a fabricated percentage in a server.

Result spans are borrowed until request closure. Errors use caller-owned records;
no cross-thread dependence on `sq_last_error`. Native diagnostics currently expose
normalized statuses rather than a reliable provider-specific numeric code:
provider NONE means unavailable, not CUDA success. Secondary reset failures and
poisoned/partial-output states are explicit. Never automatically retry a partial
response. All API error records are optional and otherwise must be initialized.

## Cache scope and security

An empty scope disables cross-request reuse. Nonempty opaque scopes (up to 64
bytes) are copied from the authenticated server, not accepted as trusted user
claims. A scope change invalidates state; identical scope still uses the existing
exact-token/state validity rules. Cancellation/errors invalidate reusable state.
One scope is cached at a time. Prefix snapshots use owned host memory and can fail
host budget checks. A scope is neither authentication nor encryption.

HTTP/SSE parsing, authentication, tenants, model aliases, queue fairness, deadlines,
metrics and protocol schemas belong in the reusable server. Its effective feature
manifest is the intersection of adapter and backend capabilities. Unsupported
harness requirements (tools, structured output, penalties, logprobs, multiple
candidates, multimodal) must produce actionable errors, not silent degradation.
Actual third-party harness conformance awaits that server. The interface and
examples are Apache-2.0; dependency/model licenses remain separate.

Executed checks and native evidence: [validation report](../reference/reports/inference-abi-validation.md).
