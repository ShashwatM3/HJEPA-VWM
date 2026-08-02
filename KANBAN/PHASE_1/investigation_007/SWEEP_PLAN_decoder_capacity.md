# Sweep plan — decoder size × reconstruction weight (× latent size?)

Master design doc for **investigation_007** — the capacity-floor sweep run after
`easy-blaze-19` (option 3 = negative result). Grounded in runs `fanciful-lake-18` (opt 1)
and `easy-blaze-19` (opt 3). Execution lives in [`GUIDE.md`](GUIDE.md); KANBAN triad in
[`DESCRIPTION.md`](DESCRIPTION.md) / `OBSERVATIONS.md` / `NEXT_STEPS.md`.

**The one fact this whole sweep exists to resolve:** every reconstruction readout in both
runs is pinned at **`L_recon_* ≈ 0.60`** (relative MSE). That floor is why option 3 had
nothing to push on. The question: **what binds that floor — the loss weight, the decoder,
or the latent `c` itself?**

---

## 0. Background — the story so far (read this first if you're a fresh session)

**The model (Phase 1, v0.2).** HJEPA-VWM is a hierarchical JEPA flow world model. Phase 1
trains coarse dynamics only:
- **`E`** — a *frozen* V-JEPA 2 ViT-L/16 encoder (~300M params). Turns a video clip into
  detailed features **`e`** (1024 tokens × 1024 dim ≈ 1.05M numbers). Never trained.
- **`B`** — trainable bottleneck. Compresses `e` → the **abstract latent `c`** (32 slots ×
  256 dim = 8,192 numbers). A **128:1 compression** — `c` is the small abstract summary the
  whole model is built around.
- **`F_c`** — trainable coarse flow predictor (rectified flow). Predicts the *future* latent
  `c_{t+k}` (`c_plus`) from the present `c_t` via a learned velocity field.
- **`B_EMA`** — EMA copy of `B` producing the (stop-grad) target `c_plus`.
- Anti-collapse: a **variance floor** (`lambda_var=0.5`) on `c`.

**The problem.** `c` is *underused*: `c_effective_rank ≈ 13`/256 (spec gate >60), and `F_c`
**loses to a copy baseline** (`coarse_vs_copy_ratio` ~1.4–3 vs gate ≤0.70) — predicting "`c`
doesn't change" beats the model. `L_flow + L_var` give `c` no reason to be information-rich.

**The reconstruction-anchor idea (investigation_006).** Add a decoder **`D`** that rebuilds
`e` from `c` with an MSE penalty — if you can rebuild the detail from the tiny summary, the
summary can't be empty. Variants: **Option 1** (decode `c_t→e_t`, grad into `D`+`B`);
**Option 2** (the brief's detached version — rejected); **Option 3** (decode *predicted*
`c_hat→e_{t+k}`, grad **through `F_c`** — the tech-lead's VITA-style joint objective).

**What investigation_006 found:**
- `fanciful-lake-18` (option 1): **removed the optimization cliff** that killed earlier runs
  (stability win) but did **not** break the rank ceiling (~13.3); copy gate still failed.
  Present-anchored recon was **blind to prediction error** (`L_recon_chat ≈ L_recon_cplus`).
- `easy-blaze-19` (option 3): **negative result.** Routing recon through `F_c` gave **no
  prediction gain** and mildly hurt the representation. Diagnosis: every reconstruction readout
  sits pinned at **~0.60** — a **capacity floor** (`c` rebuilds only ~40% of `e`'s variance),
  so a good and a bad prediction reconstruct identically and the objective can't see prediction
  quality. Training-through-`F_c` can't beat a structural floor. (Cheap-vs-clean `c_hat` is moot.)

**Why this sweep.** The floor is the gating fact. Before doing anything else with
reconstruction we must know **what binds it**: the loss **weight**, the **decoder** size, or
the **latent `c`** size (my bet). This sweep varies all three to find the binding constraint,
on a 6–8 GPU pod so it finishes in ~one run's wall-clock. Details, the failure-mode safety
analysis, and execution follow.

---

## 1. What the two terms mean, and why you (rightly) named them

### (A) Reconstruction loss scaling factor — `lambda_recon`

What it is: the weight on the reconstruction term in the total loss. Today `lambda_recon =
0.05` while `L_flow` carries weight `1.0`, so reconstruction contributes only ~5% of the
relative pull — exactly the point I made earlier.

Why sweep it: maybe the floor is **weight-bound** — we're simply not pushing reconstruction
hard enough to force `c` to encode more. If cranking the weight drops `L_recon_present`
below ~0.55 and lifts `c_effective_rank`, the floor was about *pressure*, not *capacity*,
and that's a cheap fix (no architecture change). The cost to watch: reconstruction trades
against `L_flow`; past some weight, prediction degrades. So this axis directly tests "is the
0.60 floor about how hard we're asking?"

> Note — there are actually **two** recon weights now: `lambda_recon` (present anchor, decode
> `c_t`) and `lambda_recon_pred` (option-3 prediction anchor, decode `c_hat`). For *this*
> sweep we hold `lambda_recon_pred = 0` and sweep only `lambda_recon`. Rationale: `easy-blaze-19`
> proved the prediction branch is **inert at the floor**, so adding it back now would only
> confound the diagnostic. Re-introduce option 3 *after* we confirm the floor can move.

### (B) Size of the decoder `D`

What it is: `D`'s width (`decoder_dim`, 256) and depth (`decoder_blocks`, 2) — currently
~2.17M params, mapping `c` (32×256) → `e_hat` (1024×1024).

Why you said it: the intuition that one small/thin `D` is being **pulled in two directions**
— it has to expand `c_t` → `e_t` (present) *and* `c_hat` → `e_{t+k}` (future) with one shared
set of weights. If `D` is under-parameterized for that double duty, a bigger `D` reconstructs
better and the floor drops. This is a legitimate version of the "shared-decoder tension" we
both named.

Why I pushed back earlier ("it's `c`, not `D`"): the floor looks like an **information** limit
at the latent (`c` = 8,192 numbers rebuilding `e` = ~1.05M numbers, 128:1), not a **parameter**
limit at `D`. A bigger decoder cannot recover information `c` never kept.

### Reconciling the two — and why the sweep is the only way to know

These aren't contradictory; they're two candidate binding constraints, and the floor we see
is `min(D-capacity-limit, c-information-limit, weight-limit)`. We don't know which binds
first. **That is precisely what a sweep over all three resolves.** My bet is `c`, but the
honest move is to vary each and let the floor tell us. So your instinct to keep `D` is good
(it's cheap and it disambiguates) — I just don't want `c` left untested, because it's the
variable I argued is the real bottleneck.

---

## 2. Should we also sweep the size of `c`? — My opinion: **yes, include it.**

Strong recommendation to add a latent-size axis, for three reasons:

1. **It's the hypothesis I actually hold.** I claimed the floor is `c`'s information limit.
   The only way to confirm or kill that claim is to vary `c`. A sweep over `lambda` and `D`
   that *omits* `c` leaves the most likely lever untested — we could spend 8 GPU-hours and
   still not know.
2. **It disambiguates cleanly.** If the floor drops when (and only when) `c` grows, my
   hypothesis is confirmed and the path forward is a bigger latent. If `D` or `lambda` move
   it instead, I was wrong and those are the levers — also a win.
3. **The three axes together form a diagnostic, not just a search.** See the interpretation
   matrix in §4.

**Caveats (why `c` is the "expensive/careful" axis):**
- Changing `c`'s size is **architecturally invasive**: not resume-compatible, needs fresh
  runs, fresh EMA, and it ripples through `B` (queries), `F_c` (sequence length), and `D`
  (keys). A pre-sweep audit that nothing hard-codes `32` is required (§5).
- It **changes the 128:1 compression ratio**, which is a *design pillar* — a bigger `c` is a
  less "abstract" latent and affects downstream phases. So treat `c`-size as a **diagnostic
  probe with modest values**, not a commitment to permanently enlarge `c`.
- **Use `n_c` (slot count), not `d_c`, as the knob.** Latent bandwidth = `n_c × d_c`. Bumping
  `n_c` (32 → 64) is a clean sequence-length change; bumping `d_c` (256) is deeply tied to
  `f_c_dim`, `decoder_dim`, and the variance-floor dims and is far more invasive.

---

## 3. What values to sweep (grounded in the runs)

### Per-axis ladders

| Axis | Knob | Baseline | Sweep values | Why these |
|---|---|---|---|---|
| **Recon weight** | `lambda_recon` | 0.05 | **0.1, 0.2, 0.5** | 0.5 makes recon = ½ of `L_flow`; 1.0 would equal it (recon starts dominating → too far). 4-pt ladder incl. baseline brackets "weight-bound?" |
| **Decoder size** | `decoder_dim`×`decoder_blocks` | 256×2 (~2.2M) | **512×2 (~7.5M), 512×4 (~14M)** | `D` never destabilized (`agc_D`~0.02) → headroom to grow ~3× and ~6×. Width first (most direct capacity), depth as the stronger probe. |
| **Latent size** | `n_c` | 32 (8,192 nums, 128:1) | **64 (64:1), 128 (32:1)** | Doubling/quadrupling bandwidth directly tests the information-limit hypothesis. Modest steps — diagnostic, not a commitment. |

### Don't full-grid it — staged OFAT (the efficient design)

A full 4×3×3 grid = 36 runs. Wasteful, because the *first* question is just "**which axis
moves the floor?**" — a one-factor-at-a-time scan answers that.

**Stage 1 — the binding-constraint wave (8 runs, ~one 8-GPU wave).** `fanciful-lake-18`
(λ=0.05, D=256×2, n_c=32) is the free baseline; perturb one axis at a time, plus one
"everything bigger" combo:

| # | `lambda_recon` | Decoder | `n_c` | Probes |
|---|---|---|---|---|
| 1 | 0.1 | 256×2 | 32 | weight |
| 2 | 0.2 | 256×2 | 32 | weight |
| 3 | 0.5 | 256×2 | 32 | weight (aggressive) |
| 4 | 0.05 | 512×2 | 32 | decoder width |
| 5 | 0.05 | 512×4 | 32 | decoder depth |
| 6 | 0.05 | 256×2 | 64 | latent (2×) |
| 7 | 0.05 | 256×2 | 128 | latent (4×) |
| 8 | 0.2 | 512×2 | 64 | combined (do the levers compound?) |

All with **`lambda_recon_pred = 0`** (isolate the floor question). One wave on 8 GPUs ≈ 3–4h.

**Saturation extremes (runs 9–10) — added when GPU slots are otherwise idle.** On a 5-GPU pod
the wave splits into **two passes of 5** (GUIDE §3b); rather than leave 2 cards idle in pass 2
we extend to **10 configs** with the two axis-saturation points below. They close the OFAT
blind spot the modest ladders leave open: if weight/`n_c` look *flat* at the floor, runs 1–7
can't distinguish "axis doesn't bind it" from "axis not pushed hard enough." These extremes
make that call — and they're **pre-committable** (don't depend on results), unlike an
interaction run, which can't be chosen until the binding axis is known.

| # | `lambda_recon` | Decoder | `n_c` | Probes |
|---|---|---|---|---|
| 9 | **1.0** | 256×2 | 32 | weight saturation (recon = `L_flow` weight — the ceiling of the lever) |
| 10 | 0.05 | 256×2 | **256** | `n_c` saturation (8× baseline, 32:1→16:1) — the information-limit stress test |

**Stage 2 — refine the winner (conditional, next wave).** Whatever axis moved the floor in
Stage 1, push it further and *then* re-introduce option 3 (`lambda_recon_pred = 0.05`) on the
unsaturated decoder to see if the prediction branch finally bites. If *nothing* moved the
floor → pivot away from reconstruction (see §4).

---

## 4. How to read it — two-level success + interpretation matrix

**Two gates, in order** (floor-breaking is necessary but maybe not sufficient):

1. **Did the floor move?** Watch **`L_recon_present`** (clean, no-grad). Below ~0.55 = the
   floor moved; stuck ~0.60 = it didn't.
2. **Did prediction improve?** Even if the floor drops, watch **`coarse_vs_copy_ratio`** and
   `c_effective_rank`. Floor down *and* copy-ratio toward <1 = reconstruction was the right
   lever. Floor down but copy-ratio still >2 = prediction is **not** reconstruction-bound →
   pivot to task/horizon.

**Interpretation matrix:**

| What moves `L_recon_present` below ~0.55 | Conclusion | Next |
|---|---|---|
| Higher `lambda_recon` | Floor was **weight-bound** | Cheap win — raise λ, re-test option 3 |
| Bigger `D` (and not λ/`c`) | Floor was **decoder-bound** (your intuition) | Enlarge `D`, re-test option 3 |
| Bigger `n_c` (and not λ/`D`) | Floor was **latent-capacity-bound** (my hypothesis) | Bigger `c` is the real lever |
| Nothing moves it | Floor is structural/optimization or `e` is intrinsically this incompressible | **Reconstruction is the wrong lever** — pivot to horizon/task (recall `‖Δc‖/‖c‖`~0.38 and falling → target barely moves) |

This is why the sweep is worth running even though option 3 failed: it converts "the floor
is mysterious" into a definite "lever X (or none) controls it."

---

## 4b. Will any config trigger a failure mode? — safety analysis

Short answer: **low risk across the board.** The two known failure modes are **Mode A**
(representational collapse — `c` goes uniform/identical across videos; investigation_003) and
**Mode B** (the ~step-8600 optimization cliff — `F_c` forward blow-up; investigation_005). The
reconstruction anchor is *protective* against both (it pins `c` to real per-video content and
it stabilized `F_c` in both 006 runs), so cranking it up or enlarging `D` makes collapse **less**
likely, not more. Per axis:

| Axis | Mode A (collapse) | Mode B (cliff) | Real risk to watch |
|---|---|---|---|
| Higher `lambda_recon` (0.1–0.5, **1.0**) | **lower** (anchors `c` to content) | **lower** (more `B`-stabilization) | *Soft:* at 0.5 recon competes with `L_flow` → prediction may degrade (copy ratio ↑). At the **`1.0` saturation point** recon *equals* `L_flow` and may dominate it — informative (maps the lever's ceiling) but **watch `L_flow` doesn't run away**. A measured trade-off, not a collapse; warmup ramp + abort rules cover it. |
| Bigger `D` (512×2, 512×4) | none | none | `D` never destabilized (`agc_D`~0.02); bigger = cleaner recon gradient into `B`. No collapse pathway. |
| **Bigger `n_c` (64, 128, **256**)** | **mild ↑** | low | **The one to watch.** More slots = more to keep diverse → slot-collapse risk (inv_003 territory). Also `F_c` runs on a longer token sequence (tuned for 32) — mild retune, not a hard failure. Risk scales with slot count, so **`n_c=256` (the saturation extreme) is the highest-watch run: `c_slot_diversity_rank` and `c_cross_video_cosine`.** |
| Combined #8 | low | low | Stacks low-risk changes; slightly more confounding to interpret, treat as the "best shot at the floor." |

**Safeguards already in place for every run:** the recon warmup ramp (2000 steps), the
variance floor (`lambda_var=0.5`), AGC, the post-AGC grad-skip guard, and the inv_005 abort
rules (`L_flow>1.5` sustained; rank drop >3 in 500; `agc_Fc_max_ratio` median >200). With 8
parallel runs you can't babysit each, but all log to W&B, and a config that *does* degrade is
itself informative (it tells you that config is bad) — the cost is GPU-hours, not a catastrophe.

**Bottom line:** none of the eight are *likely* to trip Mode A or Mode B — recon is protective.
The two things to glance at after the wave: `n_c=128` (slot diversity / cross-video cosine) and
the high-`lambda_recon` runs (`L_flow` not degrading). Both are visible in existing metrics.

> On the design itself: your original instinct — "each run varies one hyperparameter to one
> value" — is exactly the **OFAT** design in §3 (runs 1–7). Run 8 is the only non-OFAT entry: a
> deliberate combined "all bigger" probe, since the 8th GPU is otherwise free and it tells us
> whether the levers compound. So the table *is* your idea, plus one bonus interaction run.

---

## 5. How training will look — code changes + multi-GPU mechanics

### Code changes — DONE (committed on `phase1-v0.2-frozen-encoder`)

| Flag | Status |
|---|---|
| `--lambda-recon` / `--lambda-recon-pred` | ✅ already existed (set `--lambda-recon-pred 0` for this sweep) |
| `--decoder-dim`, `--decoder-blocks` | ✅ **added** → `cfg.model` |
| `--n-c` | ✅ **added** → `cfg.model.n_c`; **`32` hard-code audit clean** (all code reads `cfg.n_c`: `B` queries, `F_c` null-condition + slice, decoder KV is dynamic) |
| `--checkpoint-dir` | ✅ **added** → `cfg.checkpoint_dir` (mandatory per-run for parallel) |

All additive and default-to-current → the single-GPU baseline path is byte-identical. The
architecture flags (`--decoder-dim/-blocks/--n-c`) make a checkpoint **shape-incompatible**, so
**never `--resume` across different values.** `py_compile` + `ruff` clean; the option-1/option-3
gradient contracts in `smoke_test_models()` still pass on the pod.

> Full end-to-end execution (pod setup → parallel launch → monitoring), with official RunPod /
> W&B doc citations, is in [`GUIDE.md`](GUIDE.md).

### Multi-GPU execution (your 6–8× RTX PRO pod, code on the network volume)

**Approach: independent parallel runs — one config per GPU, NOT DDP.** For a sweep this is
strictly better: zero distributed-training code, and each run keeps the exact same
batch/seed/dynamics so results stay directly comparable to `fanciful`/`easy-blaze`.

Flow once the pod is up and the network volume with your repo is mounted:

```bash
# 1. Sync code (get the new flags) + sanity
cd /workspace/hierarchal-jepa-flow-world-model
git pull origin phase1-v0.2-frozen-encoder
git log -1 --oneline
python -c "from models import smoke_test_models; smoke_test_models()"   # gradient contracts

# 2. Launch the 8-wide wave in tmux (one config per GPU)
tmux new -s sweep007
mkdir -p logs
# columns: lambda  decoder_dim  decoder_blocks  n_c   (lambda_recon_pred = 0 throughout)
configs=(
  "0.1 256 2 32"
  "0.2 256 2 32"
  "0.5 256 2 32"
  "0.05 512 2 32"
  "0.05 512 4 32"
  "0.05 256 2 64"
  "0.05 256 2 128"
  "0.2 512 2 64"
)
for i in "${!configs[@]}"; do
  read L DDIM DBLK NC <<< "${configs[$i]}"
  tag="L${L}_D${DDIM}x${DBLK}_nc${NC}"
  CUDA_VISIBLE_DEVICES=$i python train.py \
    --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 --lr-coarse-flow 1e-4 \
    --lambda-recon $L --lambda-recon-pred 0 --recon-warmup-steps 2000 \
    --decoder-dim $DDIM --decoder-blocks $DBLK --n-c $NC \
    --checkpoint-dir /workspace/ckpt/$tag \
    --log-every 50 --diag-every 500 > logs/$tag.log 2>&1 &
done
wait    # all 8 finish together (~3–4h)
# Detach: Ctrl+B then D.  Reattach: tmux attach -t sweep007
```

(If the pod has 6 GPUs, use 6 configs per wave — drop two, or do `ceil(N/6)` waves.)

**Monitoring:** `nvidia-smi` (confirm all GPUs busy), `tail -f logs/<tag>.log`, and the 8 runs
appear live in W&B. Optional nicety: set `export WANDB_RUN_GROUP=sweep007` before launching so
they group on the dashboard.

**Gotchas at 8-wide:**
- **Per-run checkpoint dir is non-negotiable** (the `$tag` above). Without it, 8 runs overwrite
  `phase1_step2500.pt` simultaneously.
- **Dataloader contention:** 8 runs × `num_workers=8` = 64 workers + 8× video decode, all
  reading the dataset off the **network volume** — network IO is the likely bottleneck, and may
  cap the speedup below 8×. Mitigations: stage the dataset to local NVMe if the pod has it, drop
  `num_workers` per run (e.g. 4), or — the real fix — **cache the frozen-encoder features** so
  runs are GPU-bound and barely touch the dataset (the precompute idea from before; pays off
  most exactly here).
- **VRAM is a non-issue** — each run is a frozen encoder + ~4M trainable at batch 64; RTX PRO
  cards have far more than enough, so one run per GPU (not packing two) is the right call since
  you're compute-bound.

### The end-to-end picture

deploy pod → `git pull` (new flags) → smoke test → write the 8-config array → launch 8-wide in
tmux → ~3–4h → read `L_recon_present` + `coarse_vs_copy_ratio` per run against the §4 matrix →
identify the binding constraint → Stage 2 (push the winning axis + re-add option 3, or pivot to
horizon/task). Each completed run gets its own `investigation_007/<wandb-name>/` triad.

---

## TL;DR

- **Recon weight** tests "floor weight-bound?"; **decoder size** tests "floor decoder-bound?
  (your shared-`D`-pulled-two-ways intuition)"; **latent size** tests "floor capacity-bound?
  (my bet)." Keep all three — together they're a diagnostic, not a blind search.
- **Yes, add `n_c`** — it's the variable that decides whether my "it's `c`, not `D`" claim is
  right, and omitting it would leave the sweep unable to answer the core question. Treat it as a
  modest diagnostic probe (32→64→128), use `n_c` not `d_c`.
- **Values:** λ ∈ {0.1, 0.2, 0.5, **1.0**}; D ∈ {256×2, 512×2, 512×4}; n_c ∈ {32, 64, 128, **256**};
  `lambda_recon_pred = 0`. (The `1.0` / `n_c=256` saturation extremes are runs 9–10, added for the
  5-GPU two-wave; see §3.)
- **Design:** Stage-1 OFAT wave of 8 (one 8-GPU pass, ~3–4h) — or **10 as 5+5 two waves on a
  5-GPU pod** (~6–8h, GUIDE §3b) — to find the binding axis, then Stage 2 to refine + re-test
  option 3.
- **Code:** add `--decoder-dim`, `--decoder-blocks`, `--n-c`, and (mandatory) `--checkpoint-dir`;
  the λ flags already exist.
- **Run it** as 8 independent parallel processes pinned via `CUDA_VISIBLE_DEVICES`, per-run
  checkpoint dirs, in tmux — NOT DDP. Watch dataloader/network-IO contention.
