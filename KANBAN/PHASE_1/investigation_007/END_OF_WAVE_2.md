# END OF WAVE 2 — investigation_007 capacity-floor sweep

> **STATUS: Wave 2 produced NO usable data.** All 5 runs died at **step 200 (~350 s in)** with a
> **synchronized final heartbeat** — a whole-pod/process-group death, not a training result. The
> first diagnostic readout (step 500) never logged; the recon warmup (2000 steps) barely began.
> Section 1 is therefore a *forensic* analysis of the failed wave + a re-run prescription. Section 2
> is the real intellectual content: why we ran these waves, what Wave 1 *did* establish, the deeper
> reframe, external literature, and prioritized next steps.

Data pulled live from W&B `smahalanobis-uc-davis/hjepa-vwm`, group `inv007_capacity_floor`.
Prior context: [`SWEEP_PLAN_decoder_capacity.md`](SWEEP_PLAN_decoder_capacity.md),
[`WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md`](WAVE1_ANALYSIS_and_WAVE2_PREDICTION.md).

---

# SECTION 1 —— Wave 2 Analysis

## 1.1 What the 5 runs were supposed to be

| Run | run_id | λ_recon | Decoder | n_c | Intended probe |
|---|---|---|---|---|---|
| pious-mountain-28 | 7u5zkw6t | 0.05 | 256×2 | 64 | latent 2× (n_c) |
| classic-yogurt-29 | ryuh8cpr | 0.05 | 256×2 | 128 | latent 4× (n_c) |
| earnest-dragon-25 | 2xsd5jwr | 0.05 | 256×2 | 256 | latent 8× (n_c) — saturation |
| helpful-snow-25 | tw685b5g | 1.0 | 256×2 | 32 | weight saturation |
| quiet-firebrand-25 | bbrrydax | 0.2 | 512×2 | 64 | combined "all bigger" |

This was the latent-capacity axis (the one lever Wave 1 left untested) plus the two saturation
extremes. Per the Wave-1 writeup, **the n_c runs were the only real experiments in this wave.**

## 1.2 The finding — a synchronized death at step 200

Every run shows the identical failure signature:

| Run | runtime | last step | created | **last heartbeat** | logged history |
|---|---|---|---|---|---|
| pious-mountain-28 | 350 s | 200 | 03:41:07Z | **03:47:43Z** | step 0 only |
| classic-yogurt-29 | 381 s | 200 | 03:41:07Z | **03:47:43Z** | step 0 only |
| earnest-dragon-25 | 352 s | 200 | 03:41:07Z | **03:47:43Z** | step 0 only |
| helpful-snow-25 | 346 s | 200 | 03:41:07Z | **03:47:43Z** | step 0 only |
| quiet-firebrand-25 | 347 s | 200 | 03:41:07Z | **03:47:43Z** | step 0 only |

The smoking gun: **all five stopped reporting at the exact same second (03:47:43Z)** despite slightly
different per-process runtimes. Five independent processes do not crash on their own at the same
instant — the *pod / tmux session / process group went down together*.

**What we can rule out (so this is NOT a code or config bug):**
- **Not divergence / instability.** At the only logged point (step 0): `loss` ≈ 3.1–3.2,
  `grad_norm` ≈ 1.5–1.7, `grad_has_nan = 0`, `grad_skipped = 0`, `instability_warn = 0`. Healthy init.
- **Not an n_c shape bug.** `helpful-snow-25` (λ=1.0, n_c=32 — *identical architecture to Wave 1*)
  died at the same step as the n_c=256 run. A latent-shape bug would spare the n_c=32 run.
- **Not slow throughput.** ~1.75 s/step vs Wave 1's ~1.55 s/step (the delta is just first-200-step
  startup: encoder load, HF cache, dataloader warmup). The GPUs were working.

**What it almost certainly is:** a whole-pod event ~6 minutes after launch — the pod was
stopped/reclaimed, the tmux session was killed (e.g. SSH closed without `Ctrl-B D` detach, or the
session was terminated), the network volume dropped out from under all 5 dataloaders at once, or a
disk/OOM cascade took the process group. W&B alone can't disambiguate these — the evidence lives on
the pod, which may now be gone.

## 1.3 Why there is nothing to analyze

The diagnostic readouts (`L_recon_present`, `coarse_vs_copy_ratio`, `c_effective_rank`, …) are logged
at **diag cadence (every 500 steps)**, so the only row that exists is **step 0 — initialization**.
The step-0 numbers (`L_recon_present` ≈ 1.03 = "as bad as predicting the mean", `c_cross_video_cosine`
≈ 0.69–0.74 = the collapsed random-init value, `copy_ratio` ≈ 47–67) are **pure init**, identical in
spirit to the step-0 values of every Wave-1 run. They carry **zero signal** about the n_c hypothesis.
The recon warmup ramps over 2000 steps; the wave died at 10% of warmup.

**Net: the latent-capacity question that Wave 2 existed to answer remains open. Wave 2 must be
re-run.**

## 1.4 Re-run prescription

1. **Diagnose first (5 min):** on the pod, `tail -n 50 logs/*.log` for each tag — the tail shows the
   actual exit (Python traceback vs. a bare `Killed`/SIGTERM = external kill vs. `CUDA out of
   memory`). Check RunPod console → Pod → events for a stop/reclaim, and `dmesg | grep -i oom`.
2. **Re-launch the reduced wave** (per the Wave-1 recommendation): just **`n_c=64` + `n_c=256`** —
   the two bookends that actually decide the latent-capacity question. 2 GPUs, one short wave.
3. **Harden the launch** so a 6-minute death can't silently waste a wave again:
   - Confirm `tmux` detach (`Ctrl-B D`) **before** disconnecting SSH; verify with `tmux ls`.
   - Add an early tripwire: watch that all runs pass **step 600** (first diag log) within ~15 min
     (`grep -L "step.*500" logs/*.log` after 15 min flags any that didn't).
   - If `n_c=256` specifically OOMs, drop per-run `--batch` or `num_workers` for that run only.
4. **Cut at the plateau (~8k)** as before — but the latent runs may need slightly longer if `c`
   reorganizes, so watch `L_recon_present` flatten rather than fixing a step.

### TL;DR — Section 1
**Wave 2 is a null wave: all 5 runs died together at step 200 (~6 min), synchronized to the same
heartbeat second — a whole-pod/session death, not a training outcome.** Init was healthy (no NaN, no
instability), and the only logged data is step-0 initialization, so there is **zero signal** on the
n_c hypothesis. The latent-capacity question is still open. Re-run the reduced `n_c=64`+`n_c=256`
wave after checking the pod logs, with a step-600 tripwire so a silent early death can't recur.

---

# SECTION 2 —— Total Analysis (zoom-out)

## 2.1 Why we started these waves at all

Phase 1 trains coarse dynamics: a **frozen V-JEPA 2 ViT-L** encoder turns a clip into detailed
features `e` (~1M numbers); a trainable bottleneck `B` compresses `e` into an **abstract latent `c`**
(32 slots × 256 = 8,192 numbers, 128:1); a flow predictor `F_c` forecasts the *future* `c`. The
chronic disease across the whole project: **`F_c` loses to a copy baseline** —
`coarse_vs_copy_ratio` ≈ 1.4–3 (gate ≤ 0.70), i.e. "predict the future = the present" beats the
model — and **`c` is information-poor** (`c_effective_rank` ≈ 13/256, gate > 60).

investigation_006 tried a **reconstruction anchor**: bolt on a decoder `D: c → ê` with an MSE
penalty, on the theory that *if you can rebuild the detail from the summary, the summary can't be
empty.* Two variants ran (option 1 = anchor the present; option 3 = anchor the predicted future,
gradient through `F_c`). Both hit the same wall: every reconstruction readout pinned at
**`L_recon ≈ 0.60`**. investigation_007 (this sweep) existed to answer one question:

> **What binds that 0.60 floor — the loss weight, the decoder, or the latent `c` itself?**

Wave 1 = weight × decoder. Wave 2 = the latent `c` (n_c). Wave 2 failed to run, so the latent leg is
unanswered — **but Wave 1 alone already reframed the entire problem.**

## 2.2 What Wave 1 *definitively* established

(Full detail in the Wave-1 writeup; the three load-bearing findings:)

1. **The floor is inert to weight and decoder.** `L_recon_present` sat at **0.585 ± 0.01** across a
   5× weight range and a 6× decoder-param range. Pressure and expansion capacity are not the
   constraint.
2. **Reconstruction is structurally blind to prediction.** Decoding a *perfect* future latent
   (`L_recon_cplus`) vs `F_c`'s *predicted* one (`L_recon_chat`) differs by only **~0.01** while the
   floor is **0.585**. A perfect prediction and a bad one reconstruct almost identically → **no
   reconstruction-based objective can supervise prediction at this floor.** This is the mechanistic
   reason option 3 failed.
3. **The floor is a *utilization* limit, not a capacity one.** `c_effective_rank` held at ~13/256 in
   every run regardless of config — `c` leaves ~95% of its existing dimensions unused. The collapsed
   axis is `d_c` (per-slot dim), which **neither wave sweeps** (Wave 2's n_c adds *slots*, not dims).

## 2.3 The deeper reframe — the disease is *temporal under-informativeness*, not collapse

The crucial distinction, and the thing that points the way out:

- We are **not** representationally collapsed. `c_cross_video_cosine` falls to ~0.2 (different videos
  → different `c`) and `c_slot_diversity_rank` is healthy. The variance floor is doing its job.
- The problem is that **`c` barely changes *over time* within a trajectory.** Recall the copy
  baseline reduces *exactly* to the size of that change: `copy_loss = ‖cₜ − c₊‖²`. With
  `‖Δc‖/‖c‖ ≈ 0.38 and falling`, the target hardly moves, so "predict no change" is a brutally strong
  baseline — and `F_c` can't beat it.

So `c` is **distinct per video but nearly static in time** — it has captured *appearance/identity*
but not *dynamics*. Reconstruction makes this **worse**, not better: rebuilding `e` rewards encoding
static scene detail, which is exactly the appearance information that's already there and is *not*
what prediction needs. That's why every reconstruction lever was inert.

## 2.4 External perspectives — the literature says the same thing, louder

I researched how the field frames and fixes this. The convergence with our own data is striking:

- **"What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction"**
  ([arxiv 2606.07687](https://arxiv.org/html/2606.07687v1)) is almost a point-for-point external
  replication of our Wave-1 conclusion: reconstruction objectives *"must spend capacity on static
  scene fidelity that does not support action prediction,"* whereas temporal prediction
  *"preferentially preserves transition-relevant information."* Models with the **highest pixel
  fidelity often have near-zero action recoverability.** Their fix is an **inverse-dynamics auxiliary
  loss** (predict the transition/action from consecutive frame features) — but with a critical
  caveat: *it only **amplifies** pre-existing temporal structure* (+0.29–0.45 R² on video-pretrained
  encoders vs +0.07–0.14 on reconstruction-based ones). Good news for us: our encoder *is*
  V-JEPA 2, which already has that structure.
- **The copy/"predict-no-change" failure is a named, central JEPA failure mode.** Multiple lines
  (LeJEPA/LeWorldModel [arxiv 2603.19312](https://arxiv.org/pdf/2603.19312), the V-JEPA "temporally
  collapsed predictions" observation) describe the trivial solution where the latent goes
  (near-)constant so prediction is easy but useless. Standard defense = an anti-collapse distribution
  regularizer (SIGReg / isotropic-Gaussian); **our per-dimension variance floor is a cousin of this**
  and is why we're *not* fully collapsed — but it guards *cross-sample* variance, not *temporal*
  informativeness, which is precisely the gap we're in.
- **V-JEPA 2-AC** ([arxiv 2506.09985](https://arxiv.org/abs/2506.09985)) — the reference recipe for a
  *working* latent world model — conditions the predictor on **actions + proprioception** and trains
  with **teacher-forced next-step loss + a two-step rollout loss** that enforces multi-step
  consistency so errors don't compound.
- **Multi-step rollout losses** ([SkyJEPA 2606.23444](https://arxiv.org/html/2606.23444),
  [Persistent Robot World Models 2603.25685](https://arxiv.org/html/2603.25685)) recursively feed the
  predictor its own output to force representations that stay predictive over a horizon — a direct
  structural defense against the single-step "copy" shortcut.

**Takeaway:** the field has effectively already run our experiment and concluded *prediction, not
reconstruction, is the lever for dynamics-relevant latents.* Our Wave-1 data is an independent,
in-house confirmation.

## 2.5 Where this leaves the reconstruction thesis

investigation_006/007's reconstruction anchor is, on the evidence, **the wrong lever for the
prediction problem** — confirmed by our data (Wave 1) and corroborated by the literature. It is not
useless (it *stabilized* `F_c`, killed the optimization cliff, and pins `c` to per-video content —
all real wins), but it **cannot make `c` encode change**, and change is what `F_c` needs. Pouring more
into it (the unfinished n_c leg) is, per the Wave-1 blindness finding, very unlikely to fix
prediction even if it nudges the floor — because the floor (0.585) dwarfs the prediction signal
(~0.01) at any value n_c can realistically reach.

## 2.6 Next steps — prioritized

**Tier 0 — close the open question cheaply (so the pivot is fully evidence-backed):**
- Re-run the reduced **`n_c=64` + `n_c=256`** wave (Section 1.4). Expectation per Wave 1: floor flat
  or only cosmetically lower, `copy_ratio` still > 1, `c_effective_rank` still ~13 absolute. If so →
  all three capacity levers are formally dead and the pivot is unimpeachable. Budget: one short
  2-GPU wave.

**Tier 1 — the actual pivot: attack temporal informativeness directly.** Stop trying to make `c`
*detailed*; make it *predictive of change*. In rough order of effort/payoff:
1. **Increase the horizon `k`.** Small `Δc` is the disease, and `k` directly controls how far the
   target moves. Sweep `--horizon-k` up from 12 (e.g. 24/48); a larger gap mechanically weakens the
   copy baseline (`‖cₜ − c₊‖²` grows) and forces `c` to carry motion. Cheapest possible test of the
   reframe — no new code.
2. **Add a self-supervised inverse-dynamics / transition head.** Predict "what changed" between
   `eₜ` and `e₊` from `(cₜ, c₊)` (or from `c` directly), MSE auxiliary. This is the
   [2606.07687] recommendation, and it *amplifies* the temporal structure our V-JEPA 2 encoder
   already has. SSv2 has no robot actions, but the inter-clip transition is a free, label-free
   "action."
3. **Multi-step rollout loss on `F_c`.** Roll `F_c` forward ≥2 steps and penalize drift from the true
   future latents (with stop-grad on targets). Directly punishes the "predict no change" shortcut and
   is the V-JEPA 2-AC / SkyJEPA standard.
4. **Make the copy baseline an explicit anti-target.** Add a term that *penalizes* `ĉ` for being
   close to `cₜ` (or directly optimizes `coarse_vs_copy_ratio` margin). Higher-risk (can fight the
   variance floor) but the most direct attack on the exact metric that's failing.

**Tier 2 — only if Tier 1 stalls:** revisit the *latent dimension* `d_c` (the actually-underused
axis, rank 13/256) rather than `n_c` — but this is architecturally invasive (ripples through `B`,
`F_c`, `D`, the variance floor) and should wait until a temporal objective proves `c` *can* be driven
to higher rank at all.

**Decision gate for any Tier-1 experiment:** the only success metric that matters is
`coarse_vs_copy_ratio` trending toward < 1 (and `c_effective_rank` rising as a corroborating signal).
Reconstruction readouts are now demoted to health/diagnostics, *not* the objective.

### TL;DR — Section 2
We launched these waves to find what binds the 0.60 reconstruction floor; Wave 1 answered it
decisively — **the floor is inert to weight/decoder, reconstruction is blind to prediction, and `c`
is utilization-limited.** The deeper reframe: `c` is distinct-per-video but **nearly static in time**,
so it encodes appearance, not dynamics — and reconstruction *reinforces* appearance, which is why it
was the wrong lever. The literature (notably "Prediction over Reconstruction", V-JEPA 2-AC) reaches
the same verdict and points to **prediction-side fixes**: larger horizon, inverse-dynamics/transition
auxiliaries, and multi-step rollout losses. Next: cheaply close the n_c question via re-run, then
**pivot to making `c` predict change**, judged solely by `coarse_vs_copy_ratio` → < 1.

---

### Sources (external)
- [Prediction over Reconstruction — What Makes Video World Model Latents Action-Relevant (arXiv 2606.07687)](https://arxiv.org/html/2606.07687v1)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (arXiv 2506.09985)](https://arxiv.org/abs/2506.09985)
- [LeWorldModel: Stable End-to-End JEPA from Pixels (arXiv 2603.19312)](https://arxiv.org/pdf/2603.19312)
- [SkyJEPA: Long-Horizon World Models (arXiv 2606.23444)](https://arxiv.org/html/2606.23444)
- [Persistent Robot World Models: Stabilizing Multi-Step Rollouts (arXiv 2603.25685)](https://arxiv.org/html/2603.25685)
