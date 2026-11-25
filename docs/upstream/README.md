# squid5090

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible HTTP/SSE server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an NVIDIA RTX 5090.**

> **Upfront Attribution:** Several core Blackwell kernel designs and features in `squid5090` — including Programmatic Dependent Launch (PDL) overlap, register-permuted `mma.m16n8k16` decode GEMVs, the `128×128×64` `cp.async` MoE prefill pipeline, device-side `{2, 4, 8}` CUDA Graph conditional switch drafting (`cudaGraphCondTypeSwitch`), and the 1.8 MB runtime-gated Fisher-LDA abliteration format & tooling — were adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0). Hybrid prompt-lookup drafting was inspired by [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine), and the roofline workflow follows [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill). Full attributions live in [`NOTICE`](NOTICE).

![squid5090 RTX 5090 Benchmarks, KLD Frontier & Long-Context Scaling](assets/benchmark.png)

> **Deep Dives:**
> - **[`LITE.md`](LITE.md)** — Plain-English, 3Blue1Brown-style animated guide to GPU roofline optimization, speculative decoding, and geometric Fisher-LDA abliteration.
> - **[`LEARNINGS.md`](LEARNINGS.md)** — Complete technical reference: CUDA kernels, roofline math, KLD/quality analysis, remaining optimization headroom, and the `uv` abliteration pipeline.

---

## Benchmarks & Quality (Single RTX 5090, `c=1`)

All numbers below are measured directly on one RTX 5090 (32 GiB GDDR7) with `squid5090`'s built-in `bench`, `kld`, and `generate` commands. Every precision mode is evaluated for **exact FP64 KL divergence vs. the unmodified `exact / bf16` checkpoint** on the checked-in [`tools/kld_corpus.txt`](tools/kld_corpus.txt) (**7,784 tokens**, 7.6× past the 1,024-token local sliding window so all 8 quantized global KV layers are directly tested).

### 1. Precision × Long-Context Quality (KLD) × Decode Throughput (`mtp=7`, Adaptive `{2,4,8}` + Prompt-Lookup)

| Mode / KV | Mean KLD *(7,784 tok)* | Top-1 Match | PPL *(Ref `2.7048`)* | Code (`T=0.0` / `T=1.0`) | Code-Copy (`T=0.0` / `T=1.0`) | Chat (`T=0.0` / `T=1.0`) | Prose (`T=0.0` / `T=1.0`) | Weight VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`exact / bf16`** *(official)* | **0.0000** | 100.00% | 2.7048 | 292.8 / 248.8 tok/s | — | — | — | 18.28 GiB |
| `exact / fp8` | 0.0035 | 98.39% | 2.7007 | — | — | — | — | 18.28 GiB |
| `exact / q4` | 0.0078 | 97.15% | 2.7063 | — | — | — | — | 18.28 GiB |
| **`fp8 / fp8`** *(default ≤448K)* | **0.0061** | **97.42%** | 2.7019 | — | **402.9 / 434.9 tok/s** | 226.4 / 207.4 tok/s | 249.7 / 216.9 tok/s | 16.85 GiB |
| **`fp8 / q4`** *(default 512K)* | **0.0096** | **96.93%** | 2.7083 | **427.3 / 422.4 tok/s** | **486.8 / 433.0 tok/s** | **252.3 / 226.2 tok/s** | **239.1 / 207.6 tok/s** | 16.85 GiB |
| `nvfp4 / fp8` | 0.0230 | 94.99% | 2.7485 | — | — | — | — | 14.57 GiB |
| **`nvfp4 / q4`** *(opt-in max speed / 1M)* | **0.0263** | **94.81%** | 2.7607 | **547.0 / 485.1 tok/s** *(P0: 589.5)* | **610.5 / 518.5 tok/s** | **315.7 / 279.3 tok/s** *(P0: 332.5)* | **308.4 / 272.6 tok/s** | 14.57 GiB |

- **Accuracy-first default (`--mode fp8`):** `squid5090 serve` defaults to **`--mode fp8`** (keeping ALL 24 MoE expert layers in 100% official checkpoint precision — L0–19 NVFP4 + L20–23 MXFP8 — plus official BF16 routers/controllers, and running attention + `lm_head` through Blackwell Tensor Core `mma.m16n8k16.f32.f16` FP8 MMA + PDL). This delivers **`0.0061–0.0096` long-context KLD** (`ΔPPL -0.0030` to `+0.0035` over 7,784 tokens — **16× smaller PPL delta** than 4-bit) while reaching **`422.4 tok/s`** code (`T=1.0`, `acc=70.5%`, `dl=4.00`) and **`486.8 tok/s`** code-copy in `22.6 GiB` at 512K!
- **Temperature transparency:** `bench` defaults to `--temp 0.0` (greedy); `serve` defaults to `--temp 1.0 --top-p 0.95` (matching Inkling-2-Mini's official `generation_config.json`), using **exact Leviathan/Chen stochastic rejection sampling** (`q_i(v)` drafted from DSpark + residual resampling from $\mathrm{norm}(\max(0, p - q))$ inside the CUDA Graph).
- **Clock-state & prompt transparency:** The unified table above was measured on identical prompts at shared-GPU `P1` memory clocks (`13,801 MHz`); dedicated `P0` clocks (`14,001 MHz`) yield ~7% higher decode on the same prompts (`589.5 tok/s` `nvfp4` code, `332.5 tok/s` chat, `541.1 tok/s` ablated, `259.6 tok/s` `mtp=0`). On longer verbatim RAG/code-copy bursts (`91.5%` prompt-lookup hit rate, `P0`), `nvfp4` hybrid lookup reaches `771.6 tok/s`.
- **Short- vs. long-context KLD:** On 529-token prompts (inside the 1,024-token BF16 local window), KLD is `0.0030` (`fp8/fp8`), `0.0050` (`fp8/q4`), and `0.0184` (`nvfp4/q4`). At 7,784 tokens (7.6× past the local window, shown above), `q4` global KV adds only `+0.0035` KLD over `fp8` KV (`ΔPPL +0.0035` in `fp8/q4`).
- **Choosing a mode:** Keep the default **`--mode fp8`** (`--kv auto` selects `fp8` ≤ 448K and `q4` at 512K–1M) for near-lossless accuracy (`0.006–0.0096` KLD, `~422–487 tok/s`); opt in to **`--mode nvfp4 --kv q4`** only when you need peak 4-bit throughput (`0.0263` KLD, `ΔPPL +0.0559`) or 1M context in `23.2 GiB`.

### 2. Real Long-Context Recall vs. Synthetic 1M Step-Latency Scaling (`--mode nvfp4 --kv q4`)

Real prefilled single-needle passkey retrieval (planted at 50% prompt depth) is reported separately from synthetic zeroed-KV step latency (`squid5090 longctx`):

| Context Depth | Test Type | Prefill Latency / Rate | Decode Speed (`c=1`) | Total VRAM (0 spill) | Result / Note |
|---|---|---:|---:|---:|---|
| **20K** (`19,978` tok) | Real cold prefill | `2,282 ms` (`8,755 tok/s`) | 453.6 tok/s | 16.7 GiB | `9,181 tok/s` `B=2048` chunk peak |
| **64K** (`65,530` tok) | **Real prefilled Needle (50% depth)** | `8.46 s` (`7,742 tok/s`) | 318–331 tok/s | 16.7 GiB | **Verbatim recall PASS** |
| **128K** (`131,057` tok) | **Real prefilled Needle (50% depth)** | `21.89 s` (`5,988 tok/s`) | 166.6 tok/s | 17.4 GiB | **Verbatim recall PASS** (`COBALT-FALCON-7492`) |
| **256K** (`262,115` tok) | **Real prefilled Needle (50% depth)** | `57.29 s` (`4,575 tok/s`) | 123.9 tok/s | 17.9 GiB | **Verbatim recall PASS** (`OBSIDIAN-HERON-3815`) |
| **512K** (`523,961` tok) | **Real prefilled Needle (50% depth)** | `171.13 s` (`3,062 tok/s`) | 74.4 tok/s | 20.3 GiB | **Verbatim recall PASS** (`VERMILION-LYNX-9064`) |
| **1M** (`1,048,000` tok) | *Synthetic `longctx` (zeroed KV)* | — | **45.7 tok/s** (`B=1`, `21.86 ms/tok`)<br>*102.4 P1 / 108.0 P0 (`B=4 @dl=2.5`)* | **23.23 GiB** | Real KV allocation & step latency; `@dl=2.5` is synthetic multiplier |
| **Warm 16K / 82.5K turn** | Real HTTP prefix cache | **`71.9 ms` / `62.4 ms`** | 446.7 / 237.5 tok/s | 23.2 GiB | Host-RAM local checkpoints (`0` extra VRAM); `32 MiB` static HTTP buffer |

### 3. LoopCD-Hidden Contrastive Decoding ([`arXiv:2610.02185`](https://arxiv.org/abs/2610.02185))

Inkling-2-Mini runs its 12-layer recurrent core (`L6..L17`) for $R = 3$ passes ($h_1 \to h_2 \to h_3$) before entering the 6-layer Coda (`L18..L23`). `squid5090` implements **LoopCD-Hidden** (Eq. 2 of Liu et al., 2026) directly inside the existing Loop→Coda mega-fusion kernel (`mega_mlp_post_next_depth_norm_kernel` at visit 42) with **zero extra CUDA kernel launches**, **zero extra KV cache**, **zero CUDA Graph rebuilds** on per-request changes, and automatic prefix-cache invalidation whenever `loopcd_omega` or `loopcd_ref` changes:

$$h' = h_3 + \omega\,(h_3 - h_b), \qquad b \in \{1, 2\}$$

- **Zero-overhead & lossless verification:** When $\omega = 0.0$ (default), the kernel skips the `fmaf` cleanly (`0.000000` delta vs. baseline; `fp8/q4` KLD stays `0.009574`). When $\omega > 0$, `exact` mode retains 100% bit-exact `mtp=0 == mtp=7` parity, and `fp8/q4` (`T=1.0`, `mtp=7`) decode throughput stays equal or slightly faster (`247.5 → 260.4 tok/s` at $\omega = 0.2$, `acc: 49.2% → 52.5%`).
- **Empirical RTX 5090 sweep (`fp8/q4`, 30-task Math + Code + Logic benchmark & 7,784-tok KLD):**
  - **Thinking mode (`enable_thinking=true`, default):** $\omega \in \{0.1, 0.2\}$ with $h_b = h_1$ (and $\omega = 0.3$ with $h_2$) lifts accuracy from **`25/30 (83.3%)` → `27/30 (90.0%, +6.7 pts)`** (`10/10` Math, `10/10` Logic, `7/10` Code), with $\omega = 0.1$ using **5.5% fewer thinking tokens** (`11,893` vs `12,584`) and raising 7,784-tok top-1 match (`96.93% → 96.96%`, `ΔPPL +0.0065`). Above $\omega \ge 0.5$, generation degrades (`24/30`), matching Appendix G.1 of the paper.
  - **Direct mode (`enable_thinking=false`):** Unmodified baseline ($\omega = 0.0$) scores `29/30 (96.7%)` vs `27–28/30` at $\omega \in [0.2, 0.5]$, so `squid5090` keeps `--loopcd-omega 0.0` as the opt-in default and exposes `--loopcd-omega` / `--loopcd-ref` on both CLI and per-request HTTP JSON (`"loopcd_omega": 0.1, "loopcd_ref": 1`).

### Current Scope & Limitations
- **Text-only:** No vision/image encoder yet.
- **No constrained grammar / JSON-schema decoding** yet (standard OpenAI `tools` / `tool_calls` streaming is supported).
- **Single-user (`c=1`) slot architecture:** Optimized for lowest single-user latency and fitting up to 1M context in 23.2 GiB with `2 cudaMalloc` arenas and zero post-init host `malloc`; not a multi-user batching server.
- **HTTP request buffer:** Static `32 MiB` request & prompt-render buffers (handles 512K–1M-token JSON payloads; returns HTTP `413` above `32 MiB`).

---

## Quickstart (Linux Build & Install)

### 1. Linux Requirements

- **OS:** Linux x86_64 (tested on Ubuntu 24.04 / Debian 12–13)
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g., RTX 5090) with ≥ 16 GiB free VRAM (≥ 23.5 GiB for full 1M context)
- **NVIDIA Driver:** `580+` (tested on `590.48.01`)
- **CUDA Toolkit:** CUDA `13.0+` with `nvcc` and `cuBLAS` (tested on CUDA `13.1` at `/usr/local/cuda`)
- **Build Tools:** `g++`, `git`, `curl`, and **Zig `0.17.1+`**

```bash
sudo apt-get update && sudo apt-get install -y build-essential git curl
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version && zig version
```

*Runtime links only against `libc`, `libstdc++`, `libcudart`, `libcublas`, and `libpthread` — zero Python, CMake, Make, or external packages.*

### 2. Clone, Build & Download Weights

```bash
git clone https://github.com/tensorfish/squid5090.git
cd squid5090
zig build                     # defaults to -Doptimize=fast (ReleaseFast); binary at zig-out/bin/squid5090

uvx --from huggingface-hub hf download thinkingmachines/Inkling-2-Mini \
  --local-dir /opt/models/inkling-2-mini
```

*(Optional draft speedup: if `draft_vocab_32k.bin` — a flat little-endian `int32` array of the top 32,768 frequent token IDs — is placed in the model directory, `squid5090` automatically prunes the DSpark draft LM head from 200K to 32K for ~2× faster drafting (`868 → 428 µs`). Without it, or with `SQUID_DRAFT_VOCAB=full`, `squid5090` seamlessly uses the full 200K draft vocabulary.)*

### 3. Run the OpenAI API Server, CLI, or KLD Evaluator

```bash
# Default high-accuracy OpenAI HTTP/SSE server (FP8 Tensor Core MMA + official L0-23 MoE, 512K Q4 KV, ~22.6 GiB VRAM, 0.0096 KLD)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv q4 --ctx 524288 --mtp 7 --temp 1.0 --top-p 0.95

# Near-lossless 256K FP8 KV (0.0061 long-ctx KLD, dPPL -0.0030, ~22.6 GiB VRAM)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv fp8 --ctx 262144 --mtp 7 --temp 1.0 --top-p 0.95

# Opt-in max-speed / 1M context NVFP4 mode (~23.2 GiB at 1M, 0.0263 KLD)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --temp 1.0 --top-p 0.95

# Optional: enable 1.8 MB runtime-gated Fisher-LDA abliteration (zero weight modification)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode fp8 --kv q4 --ctx 524288 --mtp 7 \
  --ablation /opt/models/inkling-2-mini/inkling-2-mini-abliterated.safetensors

# Benchmark (supports --temp 0.0 or --temp 1.0)
./zig-out/bin/squid5090 bench /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." --mode nvfp4 --kv q4 --mtp 7 --temp 1.0 --max-tokens 1024 --runs 3

# Reproduce exact FP64 long-context KLD (7,784 tokens) vs official exact/bf16
./zig-out/bin/squid5090 logits /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --mode exact --kv bf16 --out /tmp/ref.bin
./zig-out/bin/squid5090 kld    /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --ref /tmp/ref.bin --mode fp8   --kv q4
./zig-out/bin/squid5090 kld    /opt/models/inkling-2-mini --prompt-file tools/kld_corpus.txt --raw --ctx 8192 --ref /tmp/ref.bin --mode nvfp4 --kv q4
```

**Install as a persistent `systemd` service:**
```bash
sudo cp deploy/squid5090.service /etc/systemd/system/squid5090.service
sudo systemctl daemon-reload && sudo systemctl enable --now squid5090.service
curl -s http://localhost:8000/health
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `fp8` (`serve`), `exact` (CLI) | Weight precision: `exact` (BF16 dense) \| `fp8` (FP8 TC MMA dense + official L0–23 MoE, `0.006–0.0096` KLD) \| `nvfp4` (`0.026` KLD, max speed) |
| `--kv` | `auto` | Global KV cache format: `bf16` \| `fp8` \| `q4` (enables up to `1M` context in `23.2 GiB`) |
| `--ctx` | `65536` (`serve`), `8192` (CLI) | Context window size up to `1048576` |
| `--mtp` | `7` (`serve`), `0` (CLI) | DSpark drafter (`0` = off, `7` = adaptive `{2,4,8}` CUDA Graph switch) |
| `--spec-threshold` | `0.80` | Markov confidence threshold for `{2,4,8}` verify width (`0.0` = fixed `B=8`) |
| `--temp` / `--top-p` / `--top-k` | `1.0` / `0.95` / `0` (`serve`), `0.0` (CLI) | GPU sampling & exact Leviathan/Chen rejection sampling defaults (overridable per-request) |
| `--loopcd-omega` / `--loopcd-ref` | `0.0` / `1` | Opt-in LoopCD-Hidden ([arXiv:2610.02185](https://arxiv.org/abs/2610.02185)) strength $\omega \in [-1, 2]$ (try `0.1`–`0.2` for thinking) & ref pass `1` ($h_1$) or `2` ($h_2$); overridable per HTTP request |
| `--max-tokens` | `128` (CLI), `16384` (HTTP fallback) | Max generated tokens per request |
| `--keep-history-thinking` | off | Re-inject prior turns' `reasoning_content` (off by default to prevent multi-turn thinking copy loops) |
| `--no-prompt-lookup` | off | Disable hybrid O(1) prompt-lookup drafter |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |
| `--api-key` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` auth for `serve` (constant-time compare, arbitrary key length) |

---

## Documentation & Attribution

- **[`LITE.md`](LITE.md)** — High-level, 3Blue1Brown-style animated guide to LLM optimization and abliteration.
- **[`LEARNINGS.md`](LEARNINGS.md)** — Consolidated technical reference: GPU kernels, KLD analysis, roofline audit, unexploited levers, and the `uv` abliteration pipeline.
- **[`tools/kld_corpus.txt`](tools/kld_corpus.txt)** — Checked-in 7,784-token multi-domain KLD evaluation corpus.
- **[`ledger.jsonl`](ledger.jsonl)** — Machine-readable record of every benchmark, KLD sweep, and needle run.
- **[`LICENSE`](LICENSE)** / **[`NOTICE`](NOTICE)** — Apache-2.0 license and complete open-source attributions.
