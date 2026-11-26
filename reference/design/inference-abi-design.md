# Inference ABI and reusable serving boundary

Status: design rationale. The initial experimental facade is now implemented;
see [the implemented contract](../../docs/inference-abi.md) and `include/sq_infer.h` for the
actual surface and remaining limits. The original
[reference header](abi/ifn_infer_v1_draft.h) is retained as design history, not the SDK.
Existing low-level public APIs and CUDA kernels remain unchanged.

## Purpose

Squid should expose a small inference contract that a reusable server can consume.
That server must be usable by open-source and proprietary client harnesses without
embedding Inkling-specific branches in its scheduling or transport code.

```text
Open-source / proprietary harnesses
                |
Protocol adapters: HTTP, SSE, local bindings
                |
Shared server: authentication, admission, queues, deadlines, backpressure
                |
             C inference ABI
                |
Squid backend: tokenizer, template, generation, cache, CUDA kernels
```

Harness means a client or evaluation/agent integration here. Supporting a harness
means meeting its actual protocol and semantic requirements, not merely accepting
its JSON. A closed-source harness does not require loading proprietary code into
Squid. The initial deployment should normally use HTTP between harness and server.
An embedded consumer can instead call the same C ABI directly.

## Responsibilities

| Layer | Owns |
| --- | --- |
| Harness adapter | Request/response schema, protocol version, SSE framing, protocol-specific errors and cancellation mapping |
| Shared server | Authentication, authorization, tenant identity, model aliases, admission, bounded queues, worker ownership, deadlines, disconnect handling, audit and metrics |
| Inference backend | Model validation/loading, tokenization, chat template, sampling, committed output, cache validity, cooperative cancellation, device cleanup |

The ABI contains no Keel objects, sockets, HTTP status codes, bearer tokens,
provider credentials, JSON trees or application tenants. A later Keel adapter
uses the then-current pinned Keel release. A host should be replaceable without
changing model code; a backend should be replaceable without changing transport.

The server must not claim complete OpenAI compatibility merely because it serves
a chat-completions-shaped endpoint. Responses, tools, structured output, logprobs,
multimodal input and other APIs require their own explicit support contracts.
Do not silently drop unsupported fields or reinterpret their semantics.

## Stable surface

Use `sq_infer_*` for the facade, keeping the existing experimental APIs available
for advanced diagnostics. There are three opaque owned handles:

- Model: immutable loaded weights, tokenizer and artifact identity.
- Session: prepared execution resources, context capacity and cache state.
- Request: copied input, generation configuration, cancellation and terminal result.

The sole required dynamic entry point is:

```c
sq_status sq_infer_get_api(uint32_t major, uint32_t min_minor,
                          const void **out_table);
```

It returns an immutable function table. A consumer passes a genuine `const void *`
output variable, validates negotiation, and then views it as the negotiated table
type. Static linking uses the same table. No casts between incompatible double
pointer types. No hot unloading in v1: keep the provider loaded for process lifetime.
A future GPU shared artifact should be separate from the existing host `libsq`.

The table covers discovery, defaults, model creation/planning/loading, session
creation/planning/preparation, request creation/run/cancel/progress/result and
ordered closure. Typed extensions cover backend-specific tuning. There is no
implicit worker thread or queue inside the library.

## Versioning and foreign-language consumers

The ABI is a native platform ABI, not an on-disk or wire format. Initially qualify
64-bit Linux AMD64 for CUDA execution; host-only macOS compilation is not GPU
qualification. Qualify other platforms separately.

Use fixed-width integers for discriminants/counts, IEEE-754 binary32 sampling
values, counted byte spans, opaque pointers and explicit calling conventions.
No C++ types, variadic functions, SDK structures, packed records or exported
aggregate return values. Integer constants do not depend on C enum storage size.
Allocator callbacks are a fixed-layout context/function-pointer primitive.
Callbacks are supplied as a pair, return suitably aligned memory and never throw.
The context outlives all derived handles and supports their concurrent host calls.
A NULL pair selects the provider default; allocation and release never cross owners.

Major versions may break ABI; minor versions append table entries or record tails.
Never reorder, remove or change an existing field. Unknown mandatory features fail
before execution. Negotiation failure clears the returned table pointer.

Input records begin with `struct_size, flags`. Output records begin with
`struct_size, written_size`. Initial v1 records require their complete initial
prefix. Reject invalid sizes, unknown flags and nonzero reserved fields. Bound
record sizes at 4096 bytes. A larger input tail is accepted only when zero; nonzero
unknown semantics return unsupported. Outputs preserve caller capacity, write only
known fields within it, report bytes written and leave unknown tails untouched.
Provider-owned callback events report their actual known record extent.

Extensible arrays carry count and stride; each element has its own size header.
Validate all multiplication, spans and per-element extents before access. Byte
spans and token-ID arrays have frozen primitive layouts. No NUL termination is
required for input spans. Validate UTF-8 for text and reject embedded NUL in paths.

`defaults` initializes known configuration fields within caller capacity without
allocation or execution. Defaults are greedy autoregressive decoding, no cross-
request cache reuse and no reasoning-history replay. Seed zero is a valid explicit
seed. Unbounded owned-memory budgets use `UINT64_MAX`; zero means no owned bytes.

## Lifecycle and ownership

Model and session constructors perform host bookkeeping only. Their failure returns
a NULL handle with no leaked allocations. Explicit `model_load` and
`session_prepare` acquire expensive resources. On acquisition failure, their handle
remains owned and inspectable; close releases partial resources and can be retried.
A pre-acquisition BUSY response leaves the object retryable.

Models/sessions transition CREATED → PREPARING → READY or FAILED. Fatal runtime
errors poison the session. Never reuse a poisoned execution context. A successful
close sets the caller's handle pointer to NULL; BUSY or cleanup failure retains
ownership. Parent close is BUSY while children exist. No device-wide reset.

`request_create` validates and copies input/configuration, tokenizes and prepares
bounded CPU output state. It does not mutate GPU state. Caller input buffers may
be released after success. Requests retain their session; sessions retain models.
Callback context is borrowed until `request_run` returns. Returned metadata spans
are borrowed until their owning handle is closed; copy them for longer retention.

Requests are one-shot: PREPARED → RUNNING → COMPLETED, CANCELLED or FAILED.
`request_run` is synchronous on the host-owned worker. Several prepared requests
may exist, subject to host admission limits. A competing run returns BUSY without
consuming its prepared request. There is no implicit queue or retry.

Current Squid permits **one prepared GPU session per provider process**, and one
active run. This is not a per-model limit that can be bypassed with another model
handle. Report it in discovery. The ABI must not imply continuous batching or
multi-GPU scheduling that the implementation does not provide.

Progress reads and cancellation are thread-safe on live requests. Other operations
require documented ownership serialization; closure cannot race arbitrary API
entry. Closing an active request returns BUSY. The host holds a lifetime reference
through worker completion and synchronizes before destruction.

## Cancellation correctness

Cancellation belongs to the request, not to the session. Forwarding an old request
cancel directly to `sq_session_cancel` is unsafe: it could cancel a later request.

Each request owns its atomic cancellation token. The native engine borrows that
token while holding the session execution lease and checks it at existing safe
points. Generation reset must not clear a previously accepted cancellation.
Cancelling PREPARED latches the flag without GPU work; running it then finalizes as
cancelled, or the host removes and closes it without execution.

Completion and cancellation use an atomic terminal seal. Cancellation accepted
before the seal wins: stop native work, drain outstanding GPU operations and
invalidate prefix state before publishing terminal cancellation. If completion
sealed first, cancellation reports TOO_LATE. Repeated cancellation reports
ALREADY_REQUESTED. Cleanup errors can produce FAILED instead of CANCELLED.
Hold the execution lease through drain, cache invalidation and terminal sealing.
Even cancellation during a late prefix snapshot must invalidate that snapshot.

Cancellation cannot retract streamed output and does not promise immediate kernel
preemption. A blocking user callback also delays progress. Session closure and a
new request must never race unfinished device work.

## Inputs, streaming and results

Core v1 supports typed text chat, literal raw text and optional raw token input.
Chat roles are system, developer, user and assistant, with optional name/reasoning
and an explicit absent-versus-empty content marker. The backend applies its actual
template and treats user text as untrusted text, not control tokens. Inactive input
fields must be zero. Tools and multimodal messages are outside initial v1.

Common controls are output-token limit, temperature, top-p, top-k, seed, stops,
output mode, reasoning-output policy, reasoning-history policy and required
features. Validate unsupported sampling combinations rather than approximating
them. Explicit backend profiles select precision/speculation/graph behavior;
record resolved profiles in metadata/results. No implicit algorithm downgrade.

Text callbacks deliver only committed complete UTF-8 content or separately tagged
reasoning. Draft tokens and model control strings never masquerade as content.
Replacement-versus-strict UTF-8 policy needs an explicit documented default.
Raw-token mode emits committed target IDs, including model controls; v1 raw-token
input requires raw-token or discard output. String stops are incompatible with raw
output, because withheld partial stop text cannot be represented safely as IDs.

Callbacks receive borrowed spans, ordered sequence numbers and bounded event sizes
(minimum four text bytes to accommodate one UTF-8 code point). They return continue,
normal stop, or cancel. Unknown actions fail safely. Callbacks must be bounded and
must not throw/unwind/longjmp through C. A server uses a bounded writer queue and
cancels on disconnect or backpressure timeout. Backend buffering is independent
of total response length. Discard output explicitly permits a NULL callback.

Collected responses use the same event path as streaming, avoiding two competing
decoders. A terminal result is queried separately, even after transport loss.
Normalize finishes to EOS, output limit, stop string, callback stop, cancelled or
error. Counts include target tokens consumed by the output pipeline, including
control/EOS/stop-trigger tokens; speculative proposals are separate work metrics.
A textual stop can cut inside a token, so visible bytes are not token counts.

Timing fields are duration nanoseconds with validity bits. First-content timing
means actual nonempty committed content, not an empty role/header event. Reasoning
has a separate first-output time. Queue/network latency belongs to the host; run
wall time includes callback delay. Do not infer GPU timings from host clocks.

## Capability discovery and harness compatibility

Keep backend-compiled, loaded-model-supported, configuration-enabled and currently
ready states distinct. Unsupported operation is not an unhealthy GPU. Capability
information is versioned; required feature masks are checked before GPU mutation.

Initial profile identifiers are documented backend-specific strings with explicit
validation; general profile enumeration is a later additive API. Do not advertise
unimplemented profile discovery through a field that only reports resolved values.
Typed extension IDs are backend-qualified, versioned and bounded (maximum 16 per
record). Reject duplicate, unknown or incompatible requested extensions. Extensions
must define copied payload ownership; no arbitrary JSON or executable pointers.

A generic server exposes a model capability manifest derived from the backend and
its own adapter capabilities. Effective support is their intersection. Maintain a
harness matrix with exact versions and actual required operations:

| Contract | Initial policy |
| --- | --- |
| Chat/completions and SSE | Implement adapter semantics and test collected/streamed parity |
| Disconnect, timeout, explicit cancel | Map to the same request cancellation, with no stale-request effects |
| Model listing and usage | Stable host aliases, actual backend limits and documented token accounting |
| Tool/function calls and tool history | Reject until typed input and structured output contracts exist |
| JSON/schema-constrained output | Reject until a real constrained decoder or separately named validation mode exists |
| Logprobs, penalties, multiple candidates | Reject unless explicitly supported; never fabricate values |
| Multimodal, embeddings, persistent conversations | Separate capabilities/APIs; no implied support |
| Proprietary extensions | Adapter maps to supported typed semantics or returns an actionable error |

Do not add a generic JSON passthrough to appear future-proof. New shared semantics
get additive typed capabilities; backend-only tuning uses typed extensions.
Authentication, API keys, tenant quotas and safety policies remain in the server.
Treat model paths and backend extension settings as operator configuration, not
untrusted harness-controlled inputs.

Public ABI headers and integration examples should remain Apache-2.0, without a
copyleft-only dependency in the contract. That supports proprietary consumers;
separate linked dependencies and model weights retain their own licensing terms.
Compatibility does not imply permission to distribute every model or dependency.

## Isolation, memory and diagnostics

Empty cache scope disables cross-request reuse. A nonempty copied opaque scope
(maximum 64 bytes) is supplied by the authenticated host, not trusted from a client.
Derive it from principal/conversation and relevant configuration. Different scopes
invalidate before reuse; same-scope reuse still checks exact token/state identity.
A scope is neither authentication nor encryption. Initially retain one cached
scope per session. Errors/cancellation invalidate reusable state.

Memory plans separate owned host allocations, mapped file bytes, owned device
allocations, peak owned device use and vendor estimates. Unknown estimates are
marked, not reported as zero. Budgets govern owned allocations, not all RSS or
CUDA-driver allocations; external contention can still cause OOM. Plan prefix
snapshot memory before enabling it. No hidden network downloads or checkpoint
code execution. Artifact identities distinguish verified hashes from display IDs.

Caller-owned error records include primary status, operation stage, provider code,
secondary cleanup failure and poisoned/cleanup-pending/output-started flags.
Preserve the first real failure. Cleanup GPU errors override ordinary cancellation
as the terminal failure. Do not transport errors via thread-local storage across
workers. No unsolicited prompt/credential/path-content logging, process exits,
abort or global device resets. Never blindly retry a partial streamed response.

## Implementation and acceptance sequence

1. Approve this contract and review the reference header before freezing names/layouts.
2. Implement the facade with existing native APIs; retain existing callers unchanged.
3. Implement request-owned native cancellation and terminal sealing, not an adapter
   around session-global cancellation.
4. Produce a versioned GPU provider artifact with an export allowlist and static
   equivalent. Keep the library loaded for process lifetime.
5. Prove an independent C/C++ consumer and an FFI consumer can use the negotiated
   table. Add a mock second provider to catch model-specific server assumptions.
6. Build the reusable server adapter and select actual harnesses for conformance.

Required regression coverage:

- ABI negotiation, older binary/newer minor, truncated outputs, tail canaries,
  unknown flags/extensions, record-array stride and checked-overflow failures.
- Allocation/acquisition/cleanup fault injection, BUSY ownership and retryable close.
- Cancellation before run, prefill, decode, final snapshot and completion seal;
  stale cancellation after a new request starts; close/progress/cancel races.
- Streaming versus collected output, Unicode boundaries, stop boundaries, EOS,
  reasoning policy, no draft leakage and coherent partial-output errors.
- Native parity against the qualified generation/cache baseline; cross-scope cache
  isolation, poisoned-session refusal and no persistent allocations after closure.
- Harness-level fixtures plus live tests for each claimed protocol/version,
  streaming/disconnect/deadline behavior, bounded queue overload and unsupported
  fields. Test local fake harnesses without requiring proprietary credentials;
  label unavailable proprietary integration tests unexecuted.

Design/header syntax checks establish consistency only. They do not prove binary
compatibility, native cancellation or third-party harness compatibility before the
implementation and corresponding tests exist.
