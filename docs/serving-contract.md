# Serving contract — design for review

Status: proposed; no HTTP server is implemented by this change.

Build a reusable C serving layer that consumes `ifn_infer_api_v1`. Inkleberry Finn
remains the model provider: weights, tokenization/templates, GPU execution,
speculative verification, prefix-state validity and numerical behavior stay there.
HTTP, authentication, admission, scheduling and network delivery belong to the
serving layer. No Keel or HTTP types enter the inference ABI. Implementation
repository/name is a packaging decision, not a reason to couple transport to the
model library; this document is the contract for that separate layer.

## Dependency and capability baseline

Design reviewed against [Keel v3.5.0](https://github.com/artalis-io/keel/releases/tag/v3.5.0),
commit `7035bce47302052f96fd5095e94809e828e9c0c7`. No dependency is vendored yet.
At implementation, recheck the newest stable release and pin its tag, full commit
and license. The anticipated Keel update can be adopted independently of model
mathematics and the provider ABI, after transport lifecycle tests pass.

Authoritative inspected Keel surfaces at that revision:

| Surface | Contract used by this design |
| --- | --- |
| `include/keel/async.h`, `docs/contracts/async_lifecycle.md` | One terminal per suspended operation; complete/cancel on the loop thread only. Disconnect can cancel while worker work remains outstanding. |
| `include/keel/thread_pool.h` | Work runs on a worker; done runs on the loop; never-started work has a shutdown cancel callback. Pool destruction joins workers. |
| `include/keel/http_response.h`, `http_sse.h`, `drain.h` | Streaming copies input bytes; bounded drain, accepted bytes are not delivery acknowledgments; write errors are not safely replayable frames. |
| `include/keel/http_server.h` | Suspended hang-up policy, explicit deadlines, server draining and event-context lifetime. |
| `include/keel/event_ctx.h` | Public watcher/timer APIs; context must outlive attached work and teardown callbacks. |

Use HTTP/1.1 initially, with the Linux readiness backend as the first qualification
platform. Keel's HTTP/1 streaming helpers are not HTTP/2 response helpers.
Completion-backend support needs its own executed tests before being advertised.

Negotiate ABI major 1/minor 1, validate the table extent and inspect backend,
model and session capabilities. One provider instance is loaded per process and
kept loaded until every worker and handle is retired. No duplicate loads to bypass
resource limits. The current backend supports one prepared session and one active
run per process; independent request streams are not parallel GPU execution.
Keep model artifacts external and require the operator's verified artifact manifest
for production configuration. Default precision, templates, sampling and drafting
remain the provider's; transport must not silently substitute another algorithm.

## First public protocol

| Endpoint | Behavior |
| --- | --- |
| `GET /health/live` | Event loop/control plane alive; no assertion that the GPU is making progress. |
| `GET /health/ready` | Model/session prepared, worker healthy and accepting lifecycle work; 503 while warming, poisoned or draining. Ordinary queue saturation is reported by admission, not a readiness flap. |
| `GET /v1/models` | Configured aliases and model identity; an alias never selects an arbitrary file, device or provider. |
| `POST /v1/chat/completions` | Text chat, collected JSON or SSE, single candidate. |

Advertise an OpenAI-compatible **text subset**, not universal harness support.
Support typed system/developer/user/assistant messages and names, `stream`,
`stream_options.include_usage`, temperature, top-p, seed, stops and output-token
limits through their actual ABI mappings. If both token-limit spellings are
provided they must agree. `n` must be 1. Neutral disabled options may be accepted
only when their semantics are exactly represented; unsupported non-neutral
penalties, tools, logprobs, structured-output guarantees, multimodal parts and
unknown options get actionable 400 errors. Anthropic Messages/Responses APIs and
harnesses requiring tool execution are separate milestones, not implied support.
Backend extensions use an explicit provider-qualified configuration namespace;
never pass arbitrary request JSON into the provider.

Keep reasoning opt-in and separate from answer content. Report effective features
as the intersection of protocol and provider support. Prompt/context validation
and native request creation finish before success headers are committed. Requests
may wait without headers for the bounded queue deadline; do not start a 200 SSE
response and subsequently discover an invalid prompt.

## Ownership and threads

Use one Keel event-loop thread and one explicit GPU worker, initially a
`KlThreadPool` with `num_workers=1`. Set a small explicit pool capacity and submit
only one job at a time; the bounded application admission queue owns waiting work.
Do not use automatic core-count worker defaults for GPU inference.

| Resource | Owner and lifetime |
| --- | --- |
| Provider handle and immutable function table | Process; unload after all models, sessions, requests and worker callbacks are finished. |
| Model/session and all non-cancel ABI calls | GPU worker, including load, prepare, request creation/run/result/close and teardown. Loop never tokenizes against an executing session. |
| HTTP request/response, `KlAsyncOp`, timers and transport drain | Event-loop thread; worker never reads or writes a Keel connection or response pointer. |
| Admitted input envelope | Server-owned, validated and copied before handler/body storage expires; immutable until worker completion. |
| Inference callback event | Borrowed only during callback; copy into bounded server storage before returning. |
| Output ring and copied terminal record | Job-owned until worker completion and loop retirement of all queued data/notifications. No native result span survives native request close. |
| Job registry entry | Server-owned retention record; stays alive after disconnect while worker or notifications can still reference it. |

Pin a job during any loop callback that can reenter Keel and trigger cancellation.
A transport callback clears the connection attachment before releasing that
attachment's lifetime interest. It never frees storage a worker or a queued done
callback can still access. Retire a registry entry only after compute completion,
transport retirement and notification/data retirement are all established.
Keel's exactly-one async terminal does not imply that native work has stopped.

For cancellation only, expose the native request through a short job-local handle
mutex. Worker publishes it under that mutex; the loop checks it and calls the
thread-safe `request_cancel` while holding the same lifetime guard. Before close,
the worker clears the published pointer under the mutex and then closes outside
it. A cancellation flag set before publication is observed before `request_run`.
Never hold this mutex across generation, network writes, callbacks or joining.
No stale request pointer or job ID may be redirected to the next generation.

## Scheduling and bounded resources

The loop admits raw validated envelopes; the worker creates/tokenizes a native
request only when its session is idle. This respects the provider's BUSY contract.
FIFO initially, with a per-principal active-plus-queued cap. Long running jobs are
not preempted for fairness; maximum output and execution deadlines bound their
impact. Continuous batching, token interleaving and multiple GPU workers are not
promised by this milestone.

Proposed conservative, configurable server defaults (validate before startup):

| Bound | Initial default |
| --- | --- |
| HTTP connections / body bytes | 64 connections; 1 MiB per body; explicit read/header timeouts |
| Global queued jobs / per-principal active-plus-queued | 8 / 2 |
| Retained jobs, including completed-but-draining responses | 16; completed slow clients cannot create unbounded retained state |
| Provider event bytes / encoded SSE frame | 4 KiB / 64 KiB, checked after JSON escaping and framing |
| Worker-to-loop ring / Keel drain / collected answer | 256 KiB / 256 KiB / 1 MiB per admitted job |
| Queue wait / total request deadline / stalled output | 30 s / 120 s including queue time / 5 s |
| Shutdown grace | 30 s; operator may raise it for qualified long-prefill workloads |

These are server budgets, not GPU-memory guarantees. Preserve provider defaults
for context/output tokens unless explicitly configured. Validate model/session
allocation plans and configured native budgets; account for mapped files and
unknown vendor allocations separately. Configuration ceilings do not establish
long-context qualification. Apply connection/body limits before admission as well
as queue limits after parsing; include JSON trees, copied envelopes, rings,
collected answers and notifications in server memory accounting.

Use 413 for oversized bodies, 429 for principal limits, 503 plus `Retry-After`
for global capacity/worker unavailability, and 504 for an expired queued request
whose response has not started. Reserve resources before accepting a job; failed
admission leaves no suspended operation or native work behind.

## Streaming handoff and backpressure

The native callback does bounded copy/try-enqueue work, checks cancellation and
returns promptly. It does not call Keel, allocate without bounds, wait for socket
writability or hold a GPU execution lease while blocking on a client. Full output
ring means cancel that job; the current ABI has cancellation, not resumable pause.
Keep the terminal status in a separate reserved slot so a full ring cannot lose
completion or prevent cleanup.

For the first Linux adapter use a server-owned nonblocking notification pipe and
public `KlWatcher`, separate from the output ring. Wake-ups may coalesce; the ring
and terminal flag are authoritative. Release/acquire publication plus an
armed-clear-and-recheck protocol must prevent enqueue racing wake-up consumption
from losing notification. EAGAIN on the wake pipe is not lost output. A broken
notification channel fails readiness/admission and cancels active work. Keep the
pipe, watcher and registry alive until worker and completion callbacks retire.
A future public Keel cross-thread-post API may replace this small adapter after
review; do not depend on private thread-pool wakeup fields.

The loop copies/encodes complete ordered SSE frames, enabling a finite Keel drain
before beginning the stream. Pump only while there is conservative headroom for
an entire frame plus HTTP framing. Pause pumping on a high-water mark and resume
from the loop's drain/timer notification. Bound the collected-output path too.

Keel v3.5.0's SSE/drain convenience calls do not supply a typed, retryable
whole-frame would-block result. A failed helper write can have accepted a frame
prefix: **never replay that event**. Cancel native work and close the response on
such a failure. This conservative policy works without a Keel change; typed atomic
frame admission would be a useful improvement to reassess in the upcoming release.
A slow client exceeding the output deadline is cancelled, not allowed to grow
buffers or monopolize the worker indefinitely.

Each response has one stable ID/model alias and monotonically ordered chunks.
Normal completion emits one finish/usage sequence and one `[DONE]`, then ends the
stream once. Usage comes from provider counts, not re-tokenizing text. Buffered
or socket-accepted bytes are not evidence the peer received them. Distinguish
computed tokens, server-emitted bytes and confirmed transport retirement in metrics.

## Cancellation and terminal states

Compute and transport have separate state machines:

```text
compute:   queued -> preparing -> running -> draining -> terminal
transport: attached -> sending -> finishing -> retired
                     \-> cancelled/dead -> retired
```

| Trigger | Required action |
| --- | --- |
| Disconnect while queued | Remove from admission queue; no tokenization or GPU work; retain only pending callback/notification interests. |
| Disconnect/deadline while preparing | Set the job's stop flag; cancel the native handle once published; never start generation if already abandoned. |
| Disconnect, deadline or output-ring overflow while running | Guard request lifetime and request cancellation; callback also observes the stop flag. Transport may retire before compute. |
| Completion racing cancellation | Respect the provider's ACCEPTED/TOO_LATE result and terminal seal. A late disconnect may discard a completed answer but cannot rewrite native success or cancel the next request. |
| Native run failure or failed reset/close | Preserve copied primary and cleanup errors; poison/unready the worker. No transparent replay, especially after output. |
| Deadline during an unresponsive kernel/driver call | Stop admission and mark unhealthy; cancellation is cooperative. Retain live storage until worker retirement or supervisor termination of the process. |

Retire the GPU slot only after run/drain/result/close have completed. Stream
termination is not a reason to hand the session to another job. Pending native
cleanup keeps the slot unavailable. Exactly one compute terminal and at most one
normal transport finish are allowed; neither is an exactly-once delivery claim.

Before headers, return a normal structured HTTP error. After streaming begins,
attempt one structured stream error only if the transport still has safe capacity;
otherwise close. Never append a normal finish or `[DONE]` after failure. A dead
connection receives no writes. Collected responses return an error rather than a
success-shaped partial answer. Never automatically retry after partial output.

Keel hang-up cancellation is enabled explicitly. On platforms where detection is
best effort or obscured by unread bytes, finite deadlines remain mandatory.
`on_cancel` detaches/marks dead; it does not free the worker's context or assume a
thread-pool `cancel_fn` will run for already-started work. Only the loop calls
`kl_async_complete`/`kl_async_cancel`. Completed work for a dead transport skips
resume, retires its data and releases its remaining registry interests.

## Isolation, startup and shutdown

Bind loopback by default. External access requires an explicit authentication
configuration and deployment-owned TLS/proxy policy. Principal identity comes from
verified server credentials/middleware, never a request's claimed tenant name.
Keep prompts, completions, authorization headers and keys out of ordinary logs.

Cross-request prefix reuse is initially off. Enabling it requires an explicit
server-derived isolation scope bound to authenticated principal, verified model
fingerprint, template and policy version. Never trust a client-supplied cache scope.
Scope reuse is an optimization; invalidation and exact-token validity remain the
provider's responsibility. Model aliases, request IDs and seeds are not isolation.

Startup negotiates the ABI, verifies/loads external artifacts and prepares the
single worker/session before readiness becomes 200. Record source/provider/model
identities and configured capacities, not input contents.

On shutdown: mark unready and stop admission; reject queued work; signal active
cancellation; keep driving the event loop for worker completion and output
retirement; close native request/session/model on their owner worker; join the
pool after outstanding work is retired; detach notification watchers; free Keel
server/context; unload the provider last. Pool free joins threads, so do not call
it as a cancellation mechanism on a busy or hung worker. At the finite grace
limit the supervisor may terminate the whole process; never forcibly free live
native handles, cancel a C thread, or unload a provider underneath execution.

## Acceptance and staged implementation

| Stage | Authoritative evidence before advancing |
| --- | --- |
| 1. Scheduler/lifetime core with fake provider | Barrier-controlled completion/cancel/deadline/disconnect races, same-slot ID reuse, queued cancellation never enters native code, handle publication/close race, one active run, bounded retained jobs, OOM/queue-full cleanup. |
| 2. Keel HTTP/SSE integration | Real sockets, unread-body FIN/RST, close before/after suspend, delayed worker done after transport free, wake-up coalescing/races, Unicode/stop boundaries, exact ordering, stalled readers, partial-write failure without frame replay, one normal finish, shutdown at every state. |
| 3. Real provider and harness | Exact pinned model/artifacts on an idle Blackwell; collected/streamed parity; AR/fixed/adaptive/seeded cases; native cancellation and reuse/isolation; concurrent clients; unchanged native throughput plus separately measured queue/serialization/TTFT overhead; one actual compatible external client/harness. |

Run ASan/UBSan and TSan on the deterministic CPU race suite, fuzz protocol inputs,
and repeat lifecycle tests with allocation faults. Fake GPU tests do not qualify
native behavior; real GPU benchmarks do not replace adversarial ownership tests.
Mock slow/unresponsive providers in isolated child processes for shutdown drills,
not by damaging a real CUDA context or stopping unrelated inference services.

Keep protocol tests against the installed public provider ABI, including an
unsupported-capability provider, so generic serving code cannot accidentally link
private engine structures. Record model/backend/Keel revisions and executed versus
unexecuted harness configurations. Re-run the transport suite when Keel changes;
re-run model-backed gates when inference behavior or native build/link behavior
changes. Implement stage 1 only after review of this contract.
