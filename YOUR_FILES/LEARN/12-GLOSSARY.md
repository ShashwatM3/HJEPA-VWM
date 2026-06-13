# 12 — Glossary: Every Term in One Place

> Quick lookup. Each entry: one-breath definition, then where it's taught.
> Organized by theme, not alphabet, so adjacent concepts sit together.

---

## Architecture & latents

| Term | Meaning | File |
|---|---|---|
| **World model** | A model that predicts future states of the world from current ones | 01 |
| **JEPA** | Joint-Embedding Predictive Architecture — predict the *latent* of the future/missing part, not its pixels; the encoder chooses what to ignore | 01 |
| **`e_t` (detailed latent)** | Frozen-encoder output for the context clip: (B, 1024 tokens, 1024 dims) | 02 |
| **`c_t` (abstract latent)** | Bottleneck output: (B, 32 tokens, 256 dims); the compressed "plot" of the clip | 03 |
| **`e⁺` / `c⁺` (`e_plus`/`c_plus`)** | Same latents computed on the *future* clip; `c⁺` is the prediction target (always detached) | 05 |
| **ViT-L/16** | Vision Transformer, Large (24 blocks, dim 1024), 16×16-pixel patches | 02 |
| **Tubelet** | A 2-frame × 16×16-pixel 3D patch → one token; halves temporal token count and bakes 2-frame motion into each token | 02 |
| **Temporal-major order** | Encoder token layout `index = t·256 + h·16 + w`; the assumption the bottleneck's grid reshape depends on | 02, 03 |
| **V-JEPA 2** | Meta's video encoder pretrained by masked latent prediction; our frozen `E` (`facebook/vjepa2-vitl-fpc64-256`) | 02 |
| **Frozen encoder** | All params `requires_grad=False`, eval-pinned, no_grad forward; cannot collapse, cannot adapt | 02 |
| **Bottleneck `B`** | Trainable compressor e_t → c_t: in_proj → ConvNeXt ×2 → learned-query cross-attention → residual MLP → LayerNorm | 03 |
| **ConvNeXt block** | Depthwise 7×7 conv + pointwise MLP + residual; local spatial mixing with a locality inductive bias | 03 |
| **Learned queries** | 32 free parameter vectors that cross-attend over the 1024 tokens; each learns *what to look for* (Perceiver/DETR pattern) | 03 |
| **Cross-attention vs self-attention** | Self: a set contextualizes itself (1024→1024). Cross: a small set reads from a large one (1024→32) | 03 |
| **`F_c` (coarse flow)** | 6-block DiT predicting rectified-flow velocity for the future `c`, conditioned on `c_t` | 04 |
| **DiT** | Diffusion Transformer — transformer blocks with adaLN time conditioning | 04 |
| **adaLN-Zero** | Time embedding generates per-block shift/scale/gate; gates start at zero → every block starts as identity | 04 |
| **Condition dropout** | Replace `c_t` with a learned null token 10% of training samples; robustness + classifier-free-guidance option | 04 |
| **`F_e` (fine flow)** | Phase 2: predicts future `e` conditioned on predicted `c` — the hierarchy test | 11 |
| **Shuffled-c test** | Give `F_e` the wrong clip's `c`; loss must worsen sharply or the hierarchy is decorative | 01, 11 |
| **`h_k` (horizon embedding)** | Phase 4: one learned vector per horizon k ∈ {4,8,16,32}, telling the shared `F_c` how far ahead to aim | 11 |

## Flow matching

| Term | Meaning | File |
|---|---|---|
| **Flow matching / rectified flow** | Learn a velocity field whose straight-line flow transports noise to data; train: `‖v̂ − (c⁺ − ε)‖²` | 04 |
| **`τ` (flow time)** | Position along the noise→data path, U(0,1) per sample; τ=0 pure noise, τ=1 pure data | 04 |
| **`z_τ`** | The interpolation point `(1−τ)ε + τc⁺` the network sees as input | 04 |
| **Velocity target `u`** | `c⁺ − ε`; constant along the straight path | 04 |
| **MSE blur / mean-prediction problem** | Regression to an ambiguous future returns the average of futures — sharp futures, blurry prediction; the reason for generative prediction | 01, 04 |
| **Sinusoidal/Fourier time embedding** | τ → sines/cosines at geometric frequencies so MLPs can resolve fine and coarse τ differences | 04 |

## Self-supervision & collapse

| Term | Meaning | File |
|---|---|---|
| **EMA (of weights)** | `θ_tgt ← m·θ_tgt + (1−m)·θ_online`; target = time-blurred copy of the model; horizon ≈ 1/(1−m) steps | 05 |
| **Momentum schedule** | m: 0.996 → 0.9999 (cosine over 105k); fast-tracking target early, near-frozen late | 05 |
| **`B_EMA` / TargetBottleneck** | EMA twin of the bottleneck; produces `c⁺`; never receives gradients; the only EMA module in v0.2 | 05 |
| **Stop-gradient / `as_target()`** | `.detach()` under a searchable name; gradients must never flow into targets, or the model improves loss by degrading targets | 05 |
| **Full collapse** | `c_t` constant for all inputs; loss excellent, information zero | 06 |
| **Dimensional collapse** | `c_t` varies only inside a low-dim subspace; caught by effective rank / dead-dim fraction | 06 |
| **Directional collapse** | Vectors vary in magnitude but share direction; caught by cross-video cosine | 06 |
| **Variance floor (`L_var`)** | Hinge: per-dim std across batch must reach 1.0; forbids constants, never rewards variance; λ=0.10 | 06 |
| **SIGReg / VICReg** | Covariance-decorrelation regularizers; removed in v0.2 by supervisor directive ("variance floor only, initially"); the documented escalation path for low rank | 06, 03 |

## Metrics & evaluation

| Term | Meaning | Healthy | File |
|---|---|---|---|
| **`c_std_mean/median`** | Per-dim std of `c_t` across the batch | ≈ 1.0 | 06 |
| **`c_dead_dim_frac`** | Fraction of dims with std < 10% of median | ≈ 0 | 06 |
| **`c_cross_video_cosine`** | Mean pairwise cosine of different videos' `c_t` | < 0.5 | 06 |
| **`c_effective_rank`** | exp(entropy of covariance eigenvalue distribution) — "how many dims really in use" | > 60 (Run 1: ~5) | 06 |
| **Copy baseline** | Score of pretending future = present; gate: model ≤ 0.70× | — | 06 |
| **Batch-mean baseline** | Score of predicting the average future; gate: model ≤ 0.50× (tests conditioning use) | — | 06 |
| **Acceptance gate** | A falsifiable numeric criterion that defines phase success; loss alone is never one | — | 01, 06 |
| **Goodhart's law (here)** | Add a loss that pushes a diagnostic and the diagnostic stops being evidence | — | 06 |

## Optimization

| Term | Meaning | File |
|---|---|---|
| **AdamW** | Adam (per-param adaptive steps via EMA of grad `m` and squared grad `v`) + decoupled weight decay | 07 |
| **β₂ = 0.95** | ~20-step memory for `v`; faster adaptation, thinner buffer against spikes | 07 |
| **Warmup** | Linear LR ramp (1.5k steps) while weights and Adam statistics calibrate | 07 |
| **Cosine decay** | Smooth LR decline to ~0 after warmup; big steps early, settling steps late | 07 |
| **Peak LR** | The maximum-energy moment of a run (end of warmup); where Run 1 died; "survives warmup, dies at peak → halve the peak" | 07, 10 |
| **Gradient norm** | Global L2 length of all gradients as one vector; logged *pre-clip* (shows intent) | 07 |
| **Gradient clipping** | If norm > 0.5, rescale all grads to norm 0.5; same direction, bounded step | 07 |
| **Skip-guard** | Pre-clip norm non-finite or > 50 → discard step entirely (no optimizer step, no EMA); should fire ~never | 07, 10 |
| **Gradient explosion** | Self-amplifying growth of gradient magnitudes → Inf → NaN | 10 |
| **NaN** | Result of undefined ops (Inf−Inf, 0/0); absorbing — one NaN propagates everywhere; only cure is checkpoint restore | 10 |
| **bf16** | 16-bit float: fp32's range, ~3 digits of precision; fast and overflow-proof but rounds aggressively — keep numbers small, not just finite | 07 |
| **Step / epoch** | One optimizer update / one dataset pass; 15k steps = ~240 epochs of ssv2_tiny but ~5.7 of full SSv2 — budgets only mean something relative to dataset size | 07 |
| **Overtraining/overfitting** | Too many epochs of a small set → memorization of clips instead of dynamics | 07 |

## Data & operations

| Term | Meaning | File |
|---|---|---|
| **SSv2** | Something-Something V2: ~220k short clips of object manipulation; direction-sensitive labels (no flips!) | 01, 08 |
| **ssv2_tiny** | ~4k-clip symlink subset for fast iteration | 08 |
| **VP9 / .webm** | The delta-based codec SSv2 ships in; random frame access pays keyframe-seek costs | 08 |
| **decord** | Video-decoding library; `num_threads=1` per reader (VP9 thread bug), parallelism via DataLoader workers instead | 08 |
| **Dataloader-bound** | GPU waits on CPU decode; the regime we're in (~1.4 s/step); more CPU helps, bigger GPU doesn't | 08 |
| **`horizon_k` / `frame_stride`** | Predict 4 raw frames ahead / sample context every 2nd frame | 07 |
| **Shared augmentation** | One crop + one jitter for all 16 frames of a sample; augmentation must never change the relationship being learned | 08 |
| **Stage 0** | One synthetic end-to-end train step asserting the invariants (encoder frozen, EMA moves, loss finite) before any long run | 09 |
| **tmux** | Server-side terminal session that survives SSH disconnects; all long jobs run inside one | 09 |
| **`/workspace`** | RunPod's persistent volume; everything else on a pod (pip packages, apt tools, env vars) is ephemeral | 09 |
| **`HF_HOME`** | HuggingFace cache dir; pointed at /workspace so the 1.2GB encoder downloads once | 09 |
| **W&B** | Weights & Biases — metric time series + per-run config snapshot; wrapped in try/except (logging may never kill training) | 09 |
| **Checkpoint/resume** | Every 2.5k steps: modules (incl. EMA twin) + optimizer state + step; schedules resume correctly because they're pure functions of step | 09 |
| **Postmortem** | The written autopsy of a failure; converts a $10 crash into permanent rules (POSTMORTEM_RUN1.md) | 10 |
| **Kanban / HUMAN_TASKS / PAUSE markers** | The agent–human handoff protocol: every dependency on the other actor is an explicit named pause | 09 |

---

*If a term isn't here, check `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md`
(the spec) — and consider adding it here once you've understood it.*
