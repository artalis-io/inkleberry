# `squid5090` Explained Simply (`LITE.md`)

> *An intuitive, 3Blue1Brown-style tour of how we took an LLM from **76 tokens/sec** (`exact`, `B=1`) to an accuracy-first **422–487 tokens/sec** FP8 Tensor Core default (`0.0061–0.0096` long-context KLD, `ΔPPL <= +0.0035`, `512K` context in `22.6 GiB`) — and up to **589–771 tokens/sec** / **1,000,000-token memory** in opt-in NVFP4 on one RTX 5090 — plus how to surgically remove refusals using a **1.8 MB geometric "bouncer"** with `0` diff on `53/53` tested harmless chat prompts.*
>
> **Attribution:** Key Blackwell kernel techniques (PDL overlap, Tensor Core `mma.m16n8k16` decode GEMVs, `128×128×64` MoE prefill, `{2, 4, 8}` CUDA Graph switch drafting) and the 1.8 MB runtime-gated Fisher-LDA abliteration method were adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0). Hybrid prompt-lookup drafting was inspired by [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine) (MIT). See [`NOTICE`](../../NOTICE).

---

## Part 1: Why LLMs Are Slow — And How We Made One Up to 8× Faster

### Picture a Factory With a Tiny Doorway

Imagine a giant factory (your RTX 5090 GPU). Inside, there are **thousands of hyper-fast calculators** (Tensor Cores) capable of trillions of math operations per second. Outside, in a warehouse (VRAM), sit **billions of numbers** (the model's weights).

Connecting the warehouse to the calculators is a single hallway — **memory bandwidth**. On an RTX 5090, that hallway can move about **1,513 gigabytes per second**.

Now watch what happens in the two phases of an LLM conversation:

1. **Reading your prompt (Prefill):** You hand the factory 20,000 words all at once. It brings each crate of weights through the hallway *once* and uses it on all 20,000 words. The calculators run full-blast. **Prefill is limited by math speed.**
2. **Writing the reply (Decode):** The factory can only write **one word (token) at a time**, because word #2 depends on word #1. To produce *that single word*, it has to haul **billions of weights** all the way through the hallway, do one quick multiplication per number, and throw them away — while ~90% of the calculators sit completely idle! **Decode is limited by the hallway.**

```text
  PREFILL (20,000 tokens at once — Compute-Bound)
  ┌───────────────────┐     1,513 GB/s     ┌─────────────────────────────┐
  │  VRAM Warehouse   │═══════▶  1 trip ══▶│  Tensor Cores: ██████████   │
  │  (Model Weights)  │   reused 20,000×   │  100% busy  -> 9,181 tok/s  │
  └───────────────────┘                    └─────────────────────────────┘

  DECODE (1 token at a time — Bandwidth-Bound)
  ┌───────────────────┐     1,513 GB/s     ┌─────────────────────────────┐
  │  VRAM Warehouse   │═▶ 2.8–3.7 GB/tok ─▶│  Tensor Cores: █░░░░░░░░░   │
  │  (Model Weights)  │  used ONCE, tossed │   90% IDLE  -> bottleneck!  │
  └───────────────────┘                    └─────────────────────────────┘
```

Once you see that picture, every optimization stops looking like black magic and becomes simple physics.

---

### Step 1: Measure the Hallway First (Stop Guessing)

Following the discipline from [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill), our very first move wasn't writing clever code — it was measuring the real width of the hallway (`1,513 GB/s`) and dividing by the weight bytes needed per word.

That one division tells you the **speed of light** for your GPU. If your engine is at 30% of that limit, you're wasting time on overhead. Once you're near 80–99% of that limit, no amount of micro-tweaking will help — you have to change *how many bytes* cross the hallway or *how many words* you produce per trip.

---

### Step 2: Stop Making the Manager Call the Factory 600 Times Per Word (CUDA Graphs)

Normally, generating one token requires the CPU to shout ~600 separate tiny instructions to the GPU (*"normalize! multiply! activate! attend!"*). Even if each phone call takes only a few microseconds, 600 calls per token adds up fast — especially when Python is in the middle.

- **What we did:** First, we threw away Python completely and wrote the engine in a single compiled Zig binary. Second, we recorded the entire 600-step choreography **once** into a **CUDA Graph**. Now, for every token, the CPU makes **one** call: *"replay the recording."*
- **Bonus trick (Blackwell PDL):** On the RTX 5090, we let Step $N+1$ start walking weights down the hallway *while Step $N$ is still finishing its last few calculations* — overlapping the gaps between rooms.

```text
  BEFORE (Python + eager launches): ~600 CPU calls/token + idle gaps
  CPU: [call][call][call][call]... (overhead!)
  GPU:   [Norm]   [QKV]   [Attn]   [MoE]   [LM Head]
               ^^^     ^^^      ^^^     ^^^ dead air between kernels

  AFTER (Zig + CUDA Graph + Blackwell PDL): 1 CPU call + overlapped prologues
  CPU: [Launch Graph] ──────────────────────────────────────────▶ (free!)
  GPU: [Norm][QKV][Attn][MoE][LM Head]
           └─▶ overlaps next kernel's weight fetch while current tail finishes!
```

---

### Step 3: Pack the Crates Without Breaking the Brain (FP8 Default + Opt-In NVFP4 + Q4 Memory)

If the hallway is the bottleneck, shrinking weight boxes lets you move far more weights per second — **but crushing sensitive layers to 4 bits can hurt quality if you aren't careful**. So we measured exact FP64 KL divergence across **7,784 tokens** ([`tools/kld_corpus.txt`](../tools/kld_corpus.txt), 7.6× beyond the 1,024-token local window) and built two clean precision tiers:

- **Accuracy-First Default (`--mode fp8`, `16.85 GiB`):** Leaves **all 24 MoE expert layers in 100% official checkpoint precision** (L0–19 native NVFP4 + L20–23 native MXFP8 — zero re-quantization!) and keeps routers & controllers in BF16. Only dense attention & `lm_head` move from BF16 to **FP8 E4M3** (`3.69 GB/tok`), unpacking losslessly into Blackwell Tensor Core `mma.m16n8k16.f32.f16` (exact IEEE FP32 accumulation + PDL). Result: **`0.0061` KLD** (`fp8/fp8`, `ΔPPL -0.0030`) or **`0.0096` KLD** (`fp8/q4`, `ΔPPL +0.0035`) — virtually indistinguishable from unmodified `exact/bf16`!
- **Opt-In Max-Speed / 1M (`--mode nvfp4`, `14.57 GiB`):** Packs attention, `lm_head`, and L20–23 experts into Blackwell's native 4-bit NVFP4 (`2.80 GB/tok`) for peak speed (`547–771 tok/s`) and full **1,048,576-token** context in `23.2 GiB` (`0.0263` long-ctx KLD, `ΔPPL +0.0559`).
- **512K–1M Long-Term Memory (`Q4 KV`):** Inkling-2-Mini has a hybrid design: out of 48 layer steps, **40** only look back at the last 1,024 words (tiny fixed BF16 memory!), and only **8** global visits actually store the entire conversation history (`40 + 8 = 48`). Compressing those 8 global histories into `Q4` fits **512K in `22.6 GiB` (`fp8/q4` default)** or **1,048,576 tokens in `23.2 GiB` (`nvfp4/q4`)** — with nearly 9 GiB to spare on a 32 GiB card!
- **Instant Multi-Turn Memory (Prefix Cache):** When you send Turn 2 of a 16,000-word coding session, why re-read the first 16,000 words? We snapshot the tiny local state to regular system RAM (`0` extra VRAM), so Turn 2 picks up where Turn 1 left off in **72 milliseconds** instead of 2.1 seconds.

| Precision Mode | 7,784-Tok KLD | Top-1 Match | ΔPPL *(Ref `2.7048`)* | Code (`T=0` / `T=1.0`) | Code-Copy (`T=0` / `T=1.0`) | VRAM (`ctx`) |
|---|---:|---:|---:|---:|---:|---|
| `exact / bf16` *(official ref)* | `0.0000` | 100.00% | `+0.0000` | 292.8 / 248.8 tok/s | — | 18.3 GiB *(wt)* |
| **`fp8 / fp8`** *(≤448K default)* | **`0.0061`** | **97.42%** | **`-0.0030`** | — | **402.9 / 434.9 tok/s** | **22.7 GiB** *(256K)* |
| **`fp8 / q4`** *(512K default)* | **`0.0096`** | **96.93%** | **`+0.0035`** | **427.3 / 422.4 tok/s** | **486.8 / 433.0 tok/s** | **22.6 GiB** *(512K)* |
| `nvfp4 / q4` *(opt-in max speed)* | `0.0263` | 94.81% | `+0.0559` | 547.0 / 485.1 *(P0: 589.5)* | 610.5 / 518.5 *(P0: 771.6)* | 20.3 / 23.2 GiB *(512K/1M)* |

```text
  RTX 5090 32 GiB VRAM: ACCURACY-FIRST DEFAULT (512K) vs OPT-IN 1M CONTEXT
  Default 512K (fp8 / q4, 0.0096 KLD):
  ┌─────────────────────────────┬────────────┬───────────────────────┐
  │   FP8 + Official L0-23 MoE  │ 512K Q4 KV │     Free Headroom     │
  │          16.9 GiB           │  5.7 GiB   │       ~9.4 GiB        │
  └─────────────────────────────┴────────────┴───────────────────────┘
   0                           16.9        22.6                    32.0 GiB

  Opt-In 1M (nvfp4 / q4, 0.0263 KLD):
  ┌──────────────────────────┬────────────────┬──────────────────────┐
  │   NVFP4 Model Weights    │  1M Q4 KV      │     Free Headroom    │
  │        14.6 GiB          │    8.6 GiB     │       ~8.8 GiB       │
  └──────────────────────────┴────────────────┴──────────────────────┘
   0                        14.6            23.2                    32.0 GiB

  Why 1M tokens only needs 8.6 GiB (Inkling's 5:1 Local-to-Global pattern):
  ┌──────┬──────┬──────┬──────┬──────┬────────────────────────────┐
  │ L0   │ L1   │ L2   │ L3   │ L4   │ L5  (Global Attention)     │
  │ 1024 │ 1024 │ 1024 │ 1024 │ 1024 │ Full 512K–1M Q4/FP8 KV!    │
  │ fixed│ fixed│ fixed│ fixed│ fixed│ (only 8 unique global KV)  │
  └──────┴──────┴──────┴──────┴──────┴────────────────────────────┘
```

---

### Step 4: The Real Magic Trick — Guess 7 Words Ahead for Free (Speculative Decoding)

<p align="center">
  <img src="assets/lite-speculative.gif" alt="3Blue1Brown-style animation of the RTX 5090 memory hallway and 8-token lossless speculative decoding" width="820"/>
</p>

After Steps 1–3, generating 1 token at a time hits the hallway's practical limit: **`195 tok/s`** in accuracy-first `fp8` and **`~260 tok/s`** in opt-in `nvfp4` (up from `144 tok/s` in unquantized `exact` mode). So how did we reach **`422.4 tok/s`** code (`T=1.0`) / **`486.8 tok/s`** code-copy in default `fp8/q4` — and **`589.5 / 771.6 tok/s`** (`P0, T=0`) in opt-in `nvfp4/q4`?

Remember those **~90% idle calculators** inside the factory? Checking **8 tokens at the exact same time** only requires hauling the weights through the hallway **once** — so verifying 8 words costs barely more time than generating 1 word!

So we pair the big model with **two lightning-fast "drafters"** (guessers):

1. **Drafter #1 — Prompt-Lookup (5 nanoseconds on CPU):**
   Before running any AI at all, we check a tiny hash table of the last 8,192 words: *"Have I seen these exact last 16 tokens earlier in the prompt or my own output?"*
   - When refactoring code, quoting docs, or repeating boilerplate, the answer is **yes!** We copy the next 7 tokens verbatim in nanoseconds and hand them straight to the big model to verify (**486.8 tok/s** in default `fp8/q4`, up to **771.6 tok/s** in opt-in `nvfp4`!).
2. **Drafter #2 — DSpark (0.4 milliseconds on GPU):**
   When writing brand-new code or prose, Inkling-2-Mini's built-in tiny 3-layer neural drafter predicts the next **7 tokens** *plus* a confidence score (and exact draft probability $q_i(v)$ when `temp > 0`) for each guess.
3. **Adaptive GPU Switch (`{2, 4, 8}`):**
   Here's a subtle trap: guessing 7 tokens ahead is awesome on predictable Python, but actually *slows down* unpredictable creative chat because bad guesses waste verification work.
   - Our fix: right on the GPU — without ever asking the CPU — a tiny switch checks the drafter's confidence and automatically chooses whether to verify **2, 4, or 8 tokens** this round (`cudaGraphCondTypeSwitch`), while already starting the *next* round's draft in the background!
4. **Why it's 100% lossless (even at `temp = 1.0`!):**
   - At **`temp = 0` (greedy)**, the big model grades every guessed token in parallel: if guesses 1–5 match its exact argmax and guess 6 disagrees, we keep 1–5 plus the big model's true #6 and discard 7.
   - At **`temp > 0` (server default `temp = 1.0, top_p = 0.95`)**, we run **exact Leviathan–Chen rejection sampling** right inside the CUDA Graph: accept draft token $d_i$ with probability $\alpha_i$, and on the first rejection resample from the positive residual distribution $p_{\text{res}}$:

$$\alpha_i = \min\!\left(1,\; \frac{p_{\text{target}}(d_i)}{q_{\text{draft}}(d_i)}\right), \qquad p_{\text{res}}(x) = \frac{\max\!\left(p_{\text{target}}(x) - q_{\text{draft}}(x),\; 0\right)}{\sum_y \max\!\left(p_{\text{target}}(y) - q_{\text{draft}}(y),\; 0\right)}$$

   By the law of total probability, $\mathbb{P}(\text{emit } x) \equiv p_{\text{target}}(x)$ **exactly** — mathematically identical to running the target model alone, just **2–3× faster**.

```mermaid
flowchart LR
    CTX["Last 16 Tokens"] --> PL{"CPU O(1) Prompt-Lookup<br/>(5 ns hash check)"}
    PL -- "Match found" --> VB8["GPU Verify B=8<br/>(771.6 tok/s RAG/code-copy)"]
    PL -- "No match" --> DS["GPU DSpark Drafter<br/>(3 layers, 0.4 ms, 7 guesses + conf)"]
    DS --> SW{"GPU Switch<br/>(cudaGraphCondTypeSwitch)"}
    SW -- "Low conf" --> V2["Verify B=2"]
    SW -- "Mid conf" --> V4["Verify B=4"]
    SW -- "High conf" --> V8["Verify B=8<br/>(589.5 tok/s code)"]
    V2 --> OUT["Emit Accepted + 1 Bonus Token"]
    V4 --> OUT
    V8 --> OUT
    VB8 --> OUT
    OUT -. "Next DSpark draft overlaps!" .-> DS
```

```text
  Lossless Parallel Grading (1 Weight Load -> 6 Tokens Emitted!)
  Drafted (7 guesses):  [ for ] [  i  ] [ in  ] [range] [  (  ] [ 10  ] [  )  ]
                           │       │       │       │       │       │       │
                           ▼       ▼       ▼       ▼       ▼       ▼       ▼
  Big Model Grades:     [ for ] [  i  ] [ in  ] [range] [  (  ] [  n  ] [ ... ]
                           ✓       ✓       ✓       ✓       ✓       ✗     drop
                                                                   │
  Final Output:         "for"    "i"    "in"   "range"   "("      "n"  (100% exact!)
```

---

### Step 5: Always Look Back With a Stopwatch (How Prefill Hit `9,181 tok/s`)

To make prompt-reading fast, we built a giant `128×128` Tensor Core MoE engine that processes **2,048 tokens at once** so every expert's weights are reused ~128 times. Prefill jumped from `5,077` to `7,689 tok/s`.

Then we timed every stage inside a 2,048-token chunk — and got a shock: a tiny 4-tap convolution was burning **26% of the entire runtime**! Why? Because it was running sequentially across the 2,048 tokens using only **9 out of 170 GPU cores**! Because a 4-tap convolution only looks 3 tokens backward, we sliced the 2,048 tokens into 32-token strips across **576 GPU blocks** in 30 lines of code — and prefill jumped straight to **`9,181 tok/s`** (reading a 64,000-token Needle-in-a-Haystack document in **8.46 seconds**, with verified single-needle verbatim recall PASSes all the way up through **128K in `21.9 s`**, **256K in `57.3 s`**, and **512K in `171.1 s`**).

```text
  BEFORE: 9 CTAs march serially across 2,048 tokens (161/170 SMs IDLE -> 7,689 tok/s)
  CTA 0..8: [t=0 ──────────────────────────────────────────────────────▶ t=2047]
  SMs 9..169: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ IDLE ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

  AFTER: 2D-Tiled (9 channel tiles × 64 time strips = 576 CTAs -> 9,181 tok/s!)
  Strip 0:  [t=0..31]    Strip 1:  [t=32..63]   ...   Strip 63: [t=2016..2047]
  (each strip only peeks 3 tokens left -> ALL 170 RTX 5090 SMs saturated!)
```

---

### Step 6: Free Reasoning Guidance From Discarded Loop Passes (`LoopCD-Hidden`, [arXiv:2610.02185](https://arxiv.org/abs/2610.02185))

Remember that Inkling-2-Mini is a **looped** transformer: its middle 12 layers run **3 times** ($h_1 \to h_2 \to h_3$) before handing off to the final 6 Coda layers. Normally, $h_1$ and $h_2$ get thrown away as soon as pass 3 finishes.

**LoopCD-Hidden** turns that discarded rough draft into a free steering vector right before the Coda layers:

$$h' = h_3 + \omega\,(h_3 - h_1)$$

Think of $h_1$ as the model's "first instinct" and $h_3$ as its "carefully checked answer." The difference $(h_3 - h_1)$ points straight along the direction of deeper reasoning — nudging slightly further along that arrow ($\omega \approx 0.1\text{--}0.2$) amplifies what thinking discovered while cancelling first-pass mistakes!

```text
  LoopCD-Hidden Fused at the Loop -> Coda Handoff (0 extra kernel launches, 0 extra KV!)

  Pass 1 (vi=17):  writes h_1 ─────────────────────────┐
  Pass 2 (vi=29):  writes h_2 (optional --loopcd-ref 2)│
  Pass 3 (vi=41):  computes h_3 ──▶ (+) h' = h_3 + ω(h_3 - h_1) ──▶ Coda (L18..L23) ──▶ Logits
                                     ▲ fused inside vi=42 mega-kernel!
```

Because `squid5090` fuses this directly into the existing visit-42 handoff kernel (reading $\omega$ and `ref_pass` from an 8-byte device struct), it adds **zero extra CUDA kernel launches**, **zero extra Coda KV cache**, and **zero CUDA Graph rebuilds** when toggled per HTTP request (`"loopcd_omega": 0.1`). On our 30-task RTX 5090 Math + Code + Logic benchmark in thinking mode, `--loopcd-omega 0.1` lifted accuracy from **`25/30` (83.3%) → `27/30` (90.0%)** while using **5.5% fewer thinking tokens** (and stays opt-in `--loopcd-omega 0.0` by default so baseline direct-mode behavior is untouched).

---

## Part 2: Abliteration, Explained Geometrically (3Blue1Brown Style)

Now for the second half: **what is abliteration, why do normal methods hurt Inkling-2-Mini, and how does a tiny 1.8 MB file suppress refusals while leaving `53/53` tested harmless chat prompts unchanged (`KL <= 0.03` on general text)?**

### Picture a 2,304-Dimensional Room

As Inkling-2-Mini reads a prompt and thinks through its **48 layer steps**, its internal "thought" at each step is just a point (a vector $y$) floating in a **2,304-dimensional space**.

Suppose you feed the model 336 harmless prompts (*"Explain hash maps"*, *"Write a recipe"*) and 336 prompts it normally refuses. At each step, you record where those thoughts land in the 2,304-dimensional room:
- The 336 harmless thoughts form a cloud centered at point $\mu_{\text{harmless}}$.
- The 336 refusal thoughts form a shifted cloud centered at point $\mu_{\text{harmful}}$.

Draw an arrow from the center of the harmless cloud to the center of the refusal cloud:

$$r = \text{normalize}\!\left(\mu_{\text{harmful}} - \mu_{\text{harmless}}\right)$$

That single arrow $r$ is the **"refusal direction."** Whenever the model's internal thought moves far along arrow $r$, the model outputs *"I'm sorry, I cannot assist with that."*

*(One neat detail: before finalizing $r$, we shave off any part of $r$ that points along $\mu_{\text{harmless}}$ itself — making $r$ perpendicular to typical harmless thoughts so we don't accidentally disturb normal speech.)*

---

### Why Editing the Model's Weights Fails Here

Classic abliteration permanently edits the model's weight matrices so they can *never* output anything along arrow $r$. Think of it like **putting a wooden block on the steering wheel** so the car can never turn left.

On Inkling-2-Mini, that breaks for **three** reasons:
1. **Shared looped gears:** Inkling-2-Mini has 24 physical layers, but loops the middle 12 layers **3 times** (48 logical steps). Refusal happens strongly on specific passes, while the *exact same physical weights* do Python coding and math on the other passes. Taking a hammer to a shared gear damages all 3 passes at once.
2. **4-bit quantization squashes tiny edits:** Re-packing surgical weight tweaks back into 4-bit numbers rounds them into noise.
3. **Always-on damage:** Blocking direction $r$ *even when you're just writing a Python LRU cache* nudges every word probability slightly off course.

```text
  Why Classic Weight Editing Breaks Looped Layers (24 physical -> 48 logical steps)

  Prelude         ┌───────── Shared Physical Layers 6..17 ─────────┐         Coda
  [L0..L5] ──────▶│  Pass 1 (visits 6..17)  -> reasoning & syntax  │──────▶ [L18..L23]
   6 steps        │  Pass 2 (visits 18..29) -> refusal forms here! │         6 steps
                  │  Pass 3 (visits 30..41) -> final verification  │
                  └────────────────────────────────────────────────┘
                    ▲ Classic weight edit hammers ALL 3 passes!
                    ✓ Runtime 1.8 MB sidecar targets ONLY the exact visit hill!
```

---

### The Breakthrough: A Smart 1.8 MB "Bouncer" at Runtime

<p align="center">
  <img src="assets/lite-abliteration.gif" alt="3Blue1Brown-style animation of Fisher-LDA runtime-gated geometric abliteration in 2,304-D space" width="820"/>
</p>

Instead of touching the 19 GB weights at all, we leave the model **100% untouched** and place a tiny **1.8 MB sidecar** (`--ablation`) at the exit of each of the 48 steps.

At each step $v$, when a layer produces an update vector $y$, our runtime filter does **two geometric checks**:

#### Check 1: The Bouncer — Measuring Distance in *Ellipses*, Not Circles (Fisher-LDA)
How do we know whether a thought $y$ is genuinely a refusal, and not just an unusual coding prompt?

If you just checked *"is $y$ pointing along $r$?"*, you'd get false alarms — because normal harmless text naturally stretches way out in some directions of the 2,304-dimensional room and barely moves at all in others. Picture the harmless cloud not as a neat sphere, but as a **long, squished 2,304-dimensional pancake** (described by its covariance matrix $\Sigma$).

- A step of length 1 along the *wide* direction of the pancake is completely normal harmless text.
- A step of length 1 sticking straight *out of the flat face* of the pancake is a screaming red flag!

**Fisher Linear Discriminant Analysis (Fisher-LDA)** mathematically divides out the shape of that pancake ($\Sigma^{-1} \delta$), finding the exact detector arrow $d_v$ where refusal thoughts stick farthest out of the harmless pancake. Then it scales $d_v$ so normal harmless text has a standard deviation of exactly **1**.

Now the bouncer's rule is dead simple:
> **Only intervene if $d_v \cdot y$ is more than $k$ standard deviations above normal harmless text** (typically $k \approx 2\text{--}4\sigma$).

- On normal coding, math, and chat: $y$ usually stays comfortably inside the harmless pancake → the bouncer says **"Not a refusal — pass straight through untouched!"** Output was **bit-for-bit identical on `53/53` tested harmless chat prompts** (with general-text KL drift capped at `<= 0.03` during Optuna search).
- When a refusal begins to form: $y$ pops way outside the pancake → the bouncer steps in!

```text
  2D Cross-Section of the 2,304-Dimensional Activation Room (Visit v)

      ▲ Fisher-LDA Bouncer Axis (d_v = Σ⁻¹ δ, scaled so harmless σ = 1)
      │
      │                        ★ Refusal thought y (pops > kσ out of pancake!)
      │                        │
      │                        │  Bouncer triggers! Trim ONLY excess along r_v
  ────┼────────────────────────┼─────────────────────────────────── Threshold τ_v (kσ)
      │                        ▼
      │                 ● trimmed y' (back to harmless baseline c_v)
      │
      │          ╭──────────────────────────────────────────╮
      │       ╭──╯  ·  ·  ○ coding prompt ·  ·   ·  ·       ╰──╮  Harmless Cloud
      │       ╰──╮    ·  ·   μ_harmless  ·   ·  ·  ○ chat   ╭──╯  ("squished pancake")
      │          ╰──────────────────────────────────────────╯
      │             ▲ stays BELOW τ_v -> 0 intervention (0 diff on 53/53 harmless!)
      └─────────────────────────────────────────────────────────────────────────▶
                               Wide harmless variance direction
```

#### Check 2: The Gentle Thermostat — Only Trim the *Excess*
Even when the bouncer triggers, we don't blindly zero out the whole refusal direction (normal harmless text still has a small average value $c_v$ along $r_v$). Instead, we subtract **only the excess above the harmless average**:

$$y \;\leftarrow\; y \;-\; w_v \cdot \underbrace{\mathbb{I}(d_v \cdot y > \tau_v)}_{\text{Bouncer (0 or 1)}} \cdot \underbrace{\max(r_v \cdot y - c_v,\; 0)}_{\text{Only excess above normal}} \cdot r_v$$

And because every one of the **48 logical steps** gets its own strength $w_v$ (shaped as a smooth triangular hill across the 48 steps, zero everywhere else), we only act on the exact loop passes responsible for refusal — leaving coding and reasoning passes completely alone!

```text
  Per-Visit Ablation Strength w_v Across All 48 Logical Steps
  w_v
   ▲                     ╱╲
   │                   ╱    ╲          <- Active ONLY where refusal forms
   │                 ╱        ╲
   │               ╱            ╲
   └──────────────┴──────────────┴──────────────▶ Logical Visit (0 .. 47)
     0..5 Prelude   6..17  18..29  30..41  42..47
                    Pass 1 Pass 2  Pass 3  Coda
```

---

### Step-by-Step: How `tools/ablate.py` Builds That 1.8 MB File

1. **Map the two clouds (`directions`):** Run 336 harmful and 336 harmless prompts through the reference model (once at the start of direct answers, once at the start of thinking) and compute the perpendicular refusal arrows $r_v$ at all 48 steps.
2. **Measure the harmless pancake (`stats`):** Run harmless chat, harmless reasoning, and technical prose through the model to build the `2304 × 2304` covariance matrix $\Sigma_v$ at every step, then solve for the Fisher-LDA bouncer arrows $d_v$ and harmless baselines $c_v$.
3. **Record the "perfect" harmless answers (`baseline`):** Save what the untouched model predicts on harmless prompts so we can measure *exact* intelligence drift (KL divergence) later.
4. **Let Optuna find the sweet spot (`search`):** An automated optimizer tries combinations of 10 numbers (where the triangular hill peaks across the 48 steps, how tall/wide it is, and how strict the bouncer threshold $k$ is). For each guess:
   - The reference model measures **intelligence drift** (throwing out anything that shifts harmless text by more than `0.03` KL).
   - **`squid5090 serve`** tests real refusal rates at **400–500+ tok/s** across harmful and borderline prompts!
5. **Export (`export`):** Save the winning trial's vectors and thresholds into a single **1.8 MB `.safetensors`** file — ready to plug straight into `squid5090 --ablation`.

```mermaid
flowchart LR
    D["1. directions<br/>(336 harm vs 336 harmless -> r_v)"] --> OPT
    S["2. stats<br/>(2304x2304 Σ -> Fisher-LDA d_v, c_v)"] --> OPT
    B["3. baseline<br/>(untouched harmless logits)"] --> OPT
    OPT{"4. Optuna Search<br/>(10 hill + threshold params)"} --> KL["Ref Model:<br/>KL Drift <= 0.03"]
    OPT --> EV["squid5090 serve:<br/>Refusal Eval @ 400–500+ tok/s"]
    KL --> EXP["5. export<br/>(1.8 MB .safetensors)"]
    EV --> EXP
    EXP --> SRV["squid5090 --ablation<br/>(0 diff on 53/53 harmless)"]
```

### Copy-Paste Recipe (with `uv`)

```bash
# 1. One-time setup (isolated uv env — core squid5090 still uses ZERO Python!)
uv sync --project tools
zig build                     # defaults to ReleaseFast (-Doptimize=fast)

# 2. Generate your own 1.8 MB abliterated safetensors
export SQUID_MODEL_DIR=/opt/models/inkling-2-mini
WORK=work/my-ablation && mkdir -p "$WORK"

uv run --project tools tools/ablate.py directions --out "$WORK/directions.pt"
uv run --project tools tools/ablate.py directions --out "$WORK/directions_think.pt" --think
uv run --project tools tools/ablate.py stats      --work "$WORK"
uv run --project tools tools/ablate.py baseline   --out "$WORK/baseline_val.pt" --split val
uv run --project tools tools/ablate.py baseline   --out "$WORK/baseline.pt" --split report
uv run --project tools tools/ablate.py search     --work "$WORK" --batches 12 --n 8
uv run --project tools tools/ablate.py show       --work "$WORK" --top 20
uv run --project tools tools/ablate.py export     --work "$WORK" --trial <N> --out "$WORK/my-abliterated.safetensors"

# 3. Serve with the 1.8 MB sidecar (accuracy-first fp8/q4 default: 0.0096 KLD, 512K in 22.6 GiB) and run the 32-prompt refusal check
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --mode fp8 --kv q4 --ctx 524288 --mtp 7 --temp 1.0 --top-p 0.95 --ablation "$WORK/my-abliterated.safetensors"
uv run tools/refusal_eval.py --base-url http://127.0.0.1:8000/v1 --label mine
```

---

*Want every exact formula, CUDA kernel name, and our full audit against `sybil-solutions/inference-optimisation-skill` (including what's still unimplemented)? Read [`LEARNINGS.md`](LEARNINGS.md).*
