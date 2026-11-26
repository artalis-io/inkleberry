# squid5090 — Optimization Plan & Experiment Ledger

> **Historical Archive (v0–Lever 7C):** This file preserves the original C++/CUDA bring-up notes through Lever 7C. For the current TigerStyle Zig + Blackwell engine, Levers 8–14 (PDL, TC `mma.m16n8k16`, adaptive `{2,4,8}` switch graph, `B=2048` prefill, Leviathan/Chen rejection sampling, exact FP64 KLD, and 512K needle recall), see [`README.md`](../../README.md), [`LEARNINGS.md`](LEARNINGS.md), and [`ledger.jsonl`](ledger.jsonl).

**Target:** `thinkingmachines/Inkling-2-Mini` (text + built-in DSpark MTP) on NVIDIA RTX 5090 (`sm_120a`, 32 GiB GDDR7).
**Primary Metric:** Single-user (`c=1`) decode tok/s (median of 6 runs, 512 generated tokens), with aggressive quantization + DSpark MTP enabled, verified for bitwise token parity (`mtp=0` == `mtp=K`), coherent output, and 8K/16K long-context needle retrieval.
**Reference Point:** Basalt on the same RTX 5090 logged `~261–325 tok/s` (`spec=8`, `acc=48–77%`, `dl=1.8–2.9`, using 22.9 GiB VRAM + 8.9 GiB host-RAM expert spill).

## Completed Optimization Levers

1. **v0 — Exact Native C++/CUDA Engine (`exact` mode, eager + CUDA Graph)**
   - Implemented full Inkling-2-Mini architecture in zero-dependency C++/CUDA: 48 logical visits (prelude 0..5, 3× looped core 6..17, coda 18..23), depth-attention residuals (`attn_res`), cross-pass controllers, static + dynamic short convolutions (`sconv`), L2 Q/K/R clipping, global log-position scaling, relative-bias GQA attention with learnable sinks, top-8 MoE (native NVFP4 L0–19 & MXFP8 L20–23), softcapped SwiGLU, final softcap30, and GPU argmax.
   - Eager baseline (`exact mtp=0`): **76.63 tok/s** → CUDA Graph (`exact mtp=0`): **95.83 tok/s** → `nvfp4 mtp=0`: **114.34 tok/s**.

2. **Lever 1–3 — Blackwell `sm_120a` Hardware NVFP4 PTX + All-Linear NVFP4 + Fused Projections**
   - Native `sm_120a` PTX `cvt.rn.f16x2.e2m1x2` + `half2` FMA + 128-bit `uint4` coalesced loads.
   - Quantized BF16 attention projections, LM head, controllers, router, and MXFP8 L20–23 experts to NVFP4 at load time (~2.1s direct-to-VRAM load, **14.57 GiB** weights + **0.48 GiB** runtime = **15.05 GiB** total, zero host-RAM spill).
   - Fused QKV+R+Gate+Sink projections and inline SMEM relative-bias FlashDecoding.

3. **Lever 4 — Built-in DSpark MTP Speculative Decoding + Expert-Grouped Deduplicated Verify**
   - Full DSpark 3-layer block drafter (NVFP4) + folded Markov RNN head captured inside per-`K` CUDA Graphs (`K = 1..7`).
   - Expert-grouped deduplicated MoE verify kernel (loads each unique routed expert across `B = K+1` verify tokens once).
   - `nvfp4 mtp=0`: **190.10 tok/s** | `nvfp4 mtp=3` (P1): **299.78 tok/s** median (max 317.21).

4. **Lever 5 / 5.1 — Half-Warp NVFP4 GEMV, Mega-Fused Depth/SConv Kernels, Pruned 32K Draft Vocab & Concurrent `aux_stream_` Overlap**
   - Half-warp (`16 threads/row`, 0% tail waste) `qkvrg_fused_nvfp4`, `linear_nvfp4_h`, `lm_head_stage1` (**173 µs = 1,497 GB/s = 99% of measured RTX 5090 read BW**), `dspark_base_logits` (persistent 680-block SMEM), and `markov_fold_argmax` (`MARKOV_BLOCKS=340`).
   - Vectorized `576`-thread (`18`-warp, `float4`, register-cached) `mega_attn_post_mlp_in` and `mega_mlp_post_next_depth_norm`.
   - 1-launch `mega_controller_nvfp4_kernel` (`1024` threads), parallel `<<<64, 32>>>` NVFP4 router, pruned 32K draft vocabulary (`draft_vocab_32k.bin`), and `aux_stream_` CUDA Graph fork/join overlapping `dspark_commit` (`49 µs`) with `lm_head_argmax` (`176 µs`).
   - Cut `dspark_draft` from `868 µs → 428 µs` (**2.03× faster**).
   - `nvfp4 mtp=0`: **221.39 tok/s** | `nvfp4 mtp=3` (P1): **368.49 tok/s** median (max 406.40).

5. **Lever 6 — Deterministic Zero-`atomicAdd` MoE Down + `mega_global_sconv_prep_store`**
   - Eliminated non-deterministic `atomicAdd` in `moe_down_nvfp4_grouped_h_kernel` via quarter-warp (`8 threads/row`) per-slot writes `[B, TOP_K, HIDDEN]` summed inline inside `mega_mlp_post_next_depth_norm` / `mega_mlp_post_res` (**91% peak BW** + **100% bitwise deterministic greedy parity across `mtp=0..5` and all runs**).
   - Fused all 5 kernels on the 8 global attention visits into `mega_global_sconv_prep_store_kernel` (`attn_prep_store 354 → 223 µs`).
   - Final short-context P1 median (`nvfp4 mtp=3`): **372.80 tok/s** (min 360.34, max 376.09, `acc=53.1%`, `dl=2.59`).

6. **Lever 7 (7A/7B/7C) — Maximum Context Capacity (up to 1,048,576 = 1M!) + Unified Blackwell Tensor Core (`m16n8k16` + `ldmatrix`) Split-K FlashDecoding**
   - Multi-precision global KV cache (`--kv bf16|fp8|q4`):
     - `fp8` (E4M3 + per-head FP16 scale, `1.016 B/elem`, default for `nvfp4`/`fp8`): fits up to **442,368 (432K)** tokens in **22.48 GiB** VRAM.
     - `q4` (symmetric `[-7..7]` INT4 + group-64 FP16 scale, `0.531 B/elem`, fast exact FP16 magic-bias nibble unpack) + zero-copy mapped host `embed` (`--embed-host`, auto-enabled `>512K`, frees `0.86 GiB` VRAM in full BF16): fits the **full official `1,048,576` (1M) context** in **22.47 GiB** VRAM!
   - Unified Blackwell FP16 Tensor Core (`mma.sync.aligned.m16n8k16` + `ldmatrix.x4` + `ldmatrix.x2.trans`, 2-sync `17.4 KB` SMEM) Split-K FlashDecoding across **ALL 48 visits** (64 bulk splits = 512 CTAs + 17 local/tail splits = 136 CTAs) + DSpark global layer 1 (24× HBM traffic reduction!).
   - Captured `prefill_b8_graph_exec_` (skips intermediate `lm_head_argmax`) + `--tokens-file` for arbitrarily long prompts (32K–1M+).
   - **Real end-to-end 16K decode jumped `145.67 → 338.43 tok/s` (2.32× faster!)**, **32K real decode `302.66 tok/s`**, **64K real decode `285.58 tok/s`**, short-context `mtp=0` hit a new record **`239.41 tok/s`**, and synthetic scaling holds **`202.3 tok/s` @ 256K**, **`144.4 tok/s` @ 512K**, **`115.5 tok/s` @ 768K**, and **`96.1 tok/s` @ 1,048,000 (1M)**!

## Experiment Log (All 6-Run Medians, 512 Generated Tokens unless noted)

| Run | Date | Version / Change | Mode | Prompt | `mtp` | Median tok/s | Min–Max tok/s | `acc` / `dl` | VRAM (GiB) | Verdict |
|---|---|---|---|---|---:|---:|---|---:|---:|---|
| Ref | 2026-11-23 | Basalt (`spec=8`, RAM spill) | NVFP4+Q8 | Mixed | 7 | ~261–325 | — | 48–77% / 1.8–2.9 | 22.9 (+8.9 RAM) | Reference |
| v0a | 2026-11-23 | Baseline native C++/CUDA (no graph) | `exact` | P1 | 0 | 76.63 | — | 0% / 1.00 | 18.76 | Baseline |
| v0b | 2026-11-23 | + CUDA Graph capture | `exact` | P1 | 0 | 95.83 | — | 0% / 1.00 | 18.76 | Keep |
| v0c | 2026-11-23 | + Initial NVFP4 all-linear | `nvfp4` | P1 | 0 | 114.34 | — | 0% / 1.00 | 15.05 | Keep |
| v4a | 2026-11-23 | Lever 1–4 (PTX + fusion + DSpark MTP) | `nvfp4` | P1 | 0 | 190.10 | — | 0% / 1.00 | 15.05 | Keep |
| v4b | 2026-11-23 | Lever 4 (DSpark MTP + grouped MoE) | `nvfp4` | P1 | 3 | 299.78 | —–317.21 | 58.1% / 2.73 | 15.05 | Keep |
| v5a | 2026-11-23 | Lever 5.1 (half-warp + 32K vocab + overlap) | `nvfp4` | P1 | 0 | 221.39 | —–223.33 | 0% / 1.00 | 15.05 | Keep |
| v5b | 2026-11-23 | Lever 5.1 (full 200K draft vocab A/B) | `nvfp4` | P1 | 3 | 351.45 | — | 55.4% / 2.66 | 15.05 | Reverted (32K faster) |
| v5c | 2026-11-23 | Lever 5.1 (32K draft vocab) | `nvfp4` | P1 | 3 | 368.49 | —–406.40 | 55.8% / 2.67 | 15.05 | Keep |
| v6-neg1 | 2026-11-23 | 4-launch `controller_nvfp4_batched` | `nvfp4` | profile | 3 | — | +279 µs ctrl | — | 15.05 | Reverted |
| v6-neg2 | 2026-11-23 | 512-thread `mega_controller` + `<<<32,64>>>` router | `nvfp4` | profile | 3 | — | +224 µs total | — | 15.05 | Reverted |
| v6-neg3 | 2026-11-23 | 16-lane half-warp `moe_gate_up_nvfp4` | `nvfp4` | profile | 0/3 | — | +50 / +27 µs | — | 15.05 | Reverted |
| v6-neg4 | 2026-11-23 | Lever 6.2 A/B: BF16 DSpark + NVFP4 target | `nvfp4` | P1 | 3 | 353.59 | 341.84–362.52 | 54.0% / 2.62 | 15.32 | Reverted (-13.9 tok/s) |
| **v6-P1-ex0** | 2026-11-23 | **Final Lever 6 (deterministic MoE + global mega)** | `exact` | P1 | 0 | **144.09** | 142.78–146.23 | 0% / 1.00 | 18.76 | **Final** |
| **v6-P1-ex3** | 2026-11-23 | **Final Lever 6** | `exact` | P1 | 3 | **237.56** | 232.18–240.71 | 51.0% / 2.53 | 18.76 | **Final** |
| **v6-P1-fp8-0** | 2026-11-23 | **Final Lever 6** | `fp8` | P1 | 0 | **172.95** | 169.58–175.27 | 0% / 1.00 | 17.33 | **Final** |
| **v6-P1-fp8-3** | 2026-11-23 | **Final Lever 6** | `fp8` | P1 | 3 | **282.02** | 278.25–286.82 | 47.8% / 2.43 | 17.33 | **Final** |
| **v6-P1-nv0** | 2026-11-23 | **Final Lever 6** | `nvfp4` | P1 | 0 | **220.31** | 217.03–222.22 | 0% / 1.00 | 15.05 | **Final** |
| **v6-P1-nv3** | 2026-11-23 | **Final Lever 6 (default best)** | `nvfp4` | P1 | 3 | **372.80** | 360.34–376.09 | 53.1% / 2.59 | 15.05 | **Final Best** |
| **v6-P1-nv4** | 2026-11-23 | **Final Lever 6** | `nvfp4` | P1 | 4 | **368.91** | 363.51–381.61 | 46.8% / 2.87 | 15.05 | **Final** |
| **v6-P1-nv5** | 2026-11-23 | **Final Lever 6** | `nvfp4` | P1 | 5 | **364.93** | 360.09–370.51 | 40.1% / 3.01 | 15.05 | **Final** |
| **v6-P2-nv0** | 2026-11-23 | **Final Lever 6 (Red-Black Tree code)** | `nvfp4` | P2 | 0 | **218.94** | 207.49–224.96 | 0% / 1.00 | 15.05 | **Final** |
| **v6-P2-nv3** | 2026-11-23 | **Final Lever 6 (Red-Black Tree code)** | `nvfp4` | P2 | 3 | **322.15** | 319.74–335.03 | 47.6% / 2.42 | 15.05 | **Final** |
| **v6-P3-nv0** | 2026-11-23 | **Final Lever 6 (Virtual Memory prose)** | `nvfp4` | P3 | 0 | **221.99** | 217.29–225.90 | 0% / 1.00 | 15.05 | **Final** |
| **v6-P3-nv3** | 2026-11-23 | **Final Lever 6 (Virtual Memory prose)** | `nvfp4` | P3 | 3 | **261.08** | 258.01–267.24 | 29.2% / 1.86 | 15.05 | **Final** |
| **v7c-P1-nv0** | 2026-11-23 | **Lever 7C (Unified TC `ldmatrix` + FP8 KV)** | `nvfp4/fp8` | P1 | 0 | **239.41** | 237.08–243.37 | 0% / 1.00 | 15.06 | **New B=1 Best** |
| **v7c-P1-q4** | 2026-11-23 | **Lever 7C (Unified TC `ldmatrix` + Q4 KV)** | `nvfp4/q4` | P1 | 3 | **368.59** | 357.55–378.81 | 53.0% / 2.58 | 14.92 | **Final** |
| **v7c-16K-fp8** | 2026-11-23 | **Lever 7C Real 16K Needle (`prompt=15177`)** | `nvfp4/fp8` | Needle16K | 3 | **338.43** | prefill 17.9s | 50.9% / 2.51 | 15.06 | **PASS (2.32× v6)** |
| **v7c-32K-fp8** | 2026-11-23 | **Lever 7C Real 32K Needle (`prompt=30467`)** | `nvfp4/fp8` | Needle32K | 3 | **302.66** | prefill 36.3s | 57.8% / 2.72 | 15.35 | **PASS** |
| **v7c-64K-fp8** | 2026-11-23 | **Lever 7C Real 64K Needle (`prompt=60937`)** | `nvfp4/fp8` | Needle64K | 3 | **285.58** | prefill 84.1s | 53.3% / 2.58 | 15.92 | **PASS** |
| **v7c-64K-q4** | 2026-11-23 | **Lever 7C Real 64K Needle (`prompt=60937`)** | `nvfp4/q4` | Needle64K | 3 | **278.95** | prefill 75.7s | 49.1% / 2.46 | 15.37 | **PASS** |
| **v7c-256K-q4** | 2026-11-23 | **Lever 7C Scaling (`pos=262144`, `B=4 @dl=2.5`)** | `nvfp4/q4` | `longctx` | 3 | **202.3** | 12.36 ms/step (`B=1`: 109.8 t/s) | — | 22.47 (1M alloc) | **Keep** |
| **v7c-512K-q4** | 2026-11-23 | **Lever 7C Scaling (`pos=524288`, `B=4 @dl=2.5`)** | `nvfp4/q4` | `longctx` | 3 | **144.4** | 17.31 ms/step (`B=1`: 75.9 t/s) | — | 22.47 (1M alloc) | **Keep** |
| **v7c-1M-q4** | 2026-11-23 | **Lever 7C Scaling (`pos=1048000`, `B=4 @dl=2.5`)** | `nvfp4/q4` | `longctx` | 3 | **96.1** | 26.01 ms/step (`B=1`: 45.0 t/s) | — | 22.47 (1M alloc) | **Full 1M Fits!** |

## Qualification Gates (All Passed)
- **Bitwise Token Parity (256 tok greedy):** `exact/bf16 mtp=0 == mtp=3` (**PASS**), `nvfp4/fp8 mtp=0 == mtp=3` (**PASS**), `nvfp4/q4 mtp=0 == mtp=3` (**PASS**).
- **16K Real Needle Retrieval (`prompt=15177`, `ctx=16384`):** `exact/bf16` **PASS** (`277.30 tok/s`), `nvfp4/fp8` **PASS** (`338.43 tok/s`), `nvfp4/q4` **PASS** (`323.02 tok/s`).
- **32K Real Needle Retrieval (`prompt=30467`, `ctx=32768`):** `nvfp4/fp8` **PASS** (`302.66 tok/s`), `nvfp4/q4` **PASS** (`280.96 tok/s`).
- **64K Real Needle Retrieval (`prompt=60937`, `ctx=65536`):** `nvfp4/fp8` **PASS** (`285.58 tok/s`), `nvfp4/q4` **PASS** (`278.95 tok/s`).
- **1M Context Allocation & Execution (`ctx=1048576`):** `nvfp4/q4` + zero-copy host `embed` allocates and runs cleanly in **22.47 GiB** VRAM (**96.1 tok/s @ dl=2.5** at `pos=1,048,000`).
