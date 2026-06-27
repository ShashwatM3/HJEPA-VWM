# WALK_FIXES — end-to-end pipeline audit for the SIGReg sweep

**Date:** 2026-06-27 · **Branch:** `phase1-v0.2-frozen-encoder` (committed `9f787a9`)
**Method:** trace the whole training pipeline start→finish; on any inconsistency, record
it here (fix **written, NOT applied**) and re-walk from the top; then a second dedicated
walk over **every logged metric** for stale/wrong-value bugs. Iterated until a pass found
nothing new.
**Scope audited:** `config.py`, `data.py`, `models.py`, `losses.py` (esp. `sigreg_loss`),
`diagnostics.py`, `train.py` (train loop, `train_step`, `run_diagnostics`,
`reconstruction_readouts`, logging), `tests/test_sigreg.py`, the inv008 GUIDE launch.

> **Status: nothing here BLOCKS the sweep.**
> **2026-06-27 update — F1, F2, F3 APPLIED** (+ the `run_history.py` key-list sync that F1
> implies). F6 re-estimated as safe and left as-is (see F6). F4, F5, F7, F9 deferred
> (docs/tests, low). A re-walk after these fixes found no new issues.

---

## Findings register

Severity = impact **on the inv008 conclusion** (does it corrupt `c_effective_rank` /
`coarse_vs_copy_ratio`, the decision metrics, or the SIGReg gradient that drives them?).

| # | Sev | Where | One-line |
|---|---|---|---|
| F1 | **MED** ✅ | `diagnostics.gradient_health` via `train.py:291–307,470` | `grad_global_norm` read AFTER in-place clipping → renamed `_postclip` + run_history synced |
| F2 | **MED** ✅ | `losses.sigreg_loss` called at `train.py:252` (inside autocast) | SIGReg ran in **bf16** → now forced fp32 via `autocast(enabled=False)` |
| F3 | **MED** ✅ | `train.py:252` unconditional call | global-RNG perturbation → now a **dedicated per-step generator** (byte-identical restored) |
| F4 | LOW | `losses.sigreg_loss` math | gradient can **vanish** if `c_t` scale drifts far from unit (exp saturates) |
| F5 | LOW | `losses.variance_floor` vs `sigreg_loss`/`effective_rank` | **different pooling** → the "inactive hinge" claim in SIGREG_DESIGN is approximate |
| F6 | LOW | `tests/test_sigreg.py` | absolute threshold `< 0.05` is an **unverified** estimate (could be tight) |
| F7 | LOW | `models.smoke_test_models` | **no gradient-contract test** for SIGReg (reaches B, not F_c/EMA/encoder) |
| F8 | TRIV | `train.py:252` | SIGReg's `O(P·S²)` term computed **every step** even on a λ=0 baseline |
| F9 | TRIV | `train.py:290,338` | `L_recon_pred` logged as `0.0` sentinel when option-3 off — not a quality metric |

---

## WALK 1 — the training pipeline, stage by stage

**config → CLI override → build → loop → train_step → loss → backward → AGC/clip → step →
EMA → diagnostics → log → checkpoint.**

1. **Config + CLI override (`train.py:716–779`).** ✅ Verified the architecture knobs are
   applied to `cfg.model` **before** `build_phase1_modules` and before the dataloader:
   `--decoder-dim/-blocks` (771–775), `--n-c` (776), `--checkpoint-dir` (778), `--horizon-k`
   (747), `--lambda-sigreg` (751). So the GUIDE's `--decoder-dim 512 --decoder-blocks 4`
   genuinely builds a 512×4 decoder (this was the thing that would silently wreck the
   sweep — it's wired correctly). `lr_scale` decays over `stage1_steps` (15000) = `--steps`
   (15000) for inv008, so the LR schedule is consistent — only a latent mismatch if
   `--steps ≠ 15000` (not our case). No fix.

2. **Build / EMA init (`_build_and_init`, `models.build_phase1_modules`).** ✅ B_EMA is a
   deepcopy then `copy_weights_from(B)`; encoder frozen (asserted 0 trainable). Decoder
   built at the overridden size. No fix.

3. **Train loop (`run_training:548–566`).** `val_batch = next(iter(val_loader))` is fetched
   **once** and reused for all diagnostics — ✅ correct (cross-step/cross-run comparability
   of `c_effective_rank`). `metrics = train_step(...)` is a **fresh dict each step**;
   diag keys `.update`d only when `step % diag_every == 0`; `wandb.log(metrics, step=step)`
   only when `step % log_every == 0`. Since `diag_every=500` is a multiple of `log_every=50`,
   every diag step is logged. **No stale carryover** — the specific "old value re-logged" bug
   is NOT present (each step rebuilds `metrics`; sparse keys are fine for W&B). ✅

4. **`train_step` forward (`238–290`).** `_coarse_forward` → `abstract` (grad), `target_abstract`
   (stop-grad, EMA), `detailed`/`target_detailed` (frozen, no-grad). Flow loss, var/cov/slot,
   **sigreg**, recon. `loss.backward()` (291) is correctly **outside** the autocast block. ✅
   → **but** the sigreg term itself is computed **inside** autocast → **F2**, and is called
   **unconditionally** → **F3** (both below).

5. **AGC + clip + skip + step + EMA (`292–330`).** `apply_trainable_agc` (292) and
   `clip_grad_norm_(trainable, 0.5)` (307) both mutate `.grad` **in place**; grads are NOT
   re-zeroed at the end of `train_step` (only `zero_grad` at the *start* of the next step, or
   on a skip). `grad_norm` (the value `clip_grad_norm_` returns) is the **pre-clip** total
   norm — authoritative, logged every step. ✅ → **but** `run_diagnostics` later reads the
   now-clipped grads → **F1**.

6. **`run_diagnostics` (`448–475`).** Recomputes `abstract` on the fixed `val_batch` under
   `no_grad`; all `c_*` metrics computed **fresh** on the current model, in **fp32 (outside
   autocast)**. ✅ So `c_effective_rank`, `coarse_vs_copy_ratio`, `L_recon_*` — the decision
   metrics — are precision-clean and not stale. The only diagnostic flaw is `grad_global_norm`
   (F1).

7. **Checkpoint (`save/load`).** Per-run dir (GUIDE); no `--resume` across the arch change. ✅

---

## WALK 2 — every logged metric (the "is it the value I think it is?" pass)

Legend: **src** = where computed; **batch** = train(64) or val(16); **fresh?** = recomputed
this step (not carried).

| metric | src | batch | fresh? | correct? |
|---|---|---|---|---|
| `loss,L_flow,L_var,L_cov,L_slot` | train_step | train | ✅ | ✅ raw, pre-λ (consistent convention) |
| **`L_sigreg`** | train_step:337 | train | ✅ | value ✅ (raw, pre-λ); but computed in **bf16** (F2) |
| `L_recon` | train_step:329 | train | ✅ | ✅ raw rel-MSE; real for inv008 (λ_recon=0.05>0) |
| `L_recon_pred` | train_step:338 | train | ✅ | =0.0 **sentinel** (option-3 off) — not quality (F9) |
| `recon_scale` | train_step | – | ✅ | ✅ ramps 0→1 over 2000; the actual loss weight is λ·scale·L |
| `grad_norm` | train_step:307 | train | ✅ | ✅ **pre-clip** total norm — the authoritative grad signal |
| `grad_skipped,instability_warn,ema_m,agc_*,lr_mult` | train_step/loop | – | ✅ | ✅ current-step |
| `c_std_*,c_dead_dim_frac` | variance_stats | val | ✅ | ✅ |
| `c_cross_video_cosine` | diagnostics | val | ✅ | ✅ |
| **`c_effective_rank`** | diagnostics:76 | val | ✅ | ✅ fp32; exp(entropy(eig)); same pooling as SIGReg (decision metric — clean) |
| `c_slot_diversity_rank` | diagnostics:103 | val | ✅ | ✅ |
| **`coarse_vs_copy_ratio`** + losses | coarse_baselines | val | ✅ | ✅ copy = ‖c_t−c_plus‖²; ratio<1 ⇒ beats copy (decision metric — clean) |
| `coarse_vs_batch_mean_ratio` | coarse_baselines | val | ✅ | ✅ |
| `grad_global_norm` | gradient_health:470 | – | ⚠️ | **F1** — reads **post-clip** grads → pinned ≤~0.5, misleading |
| `grad_has_nan,grad_param_count` | gradient_health | – | ✅ | ✅ (NaN survives clip, so the flag is still valid) |
| `L_recon_present/cplus/chat` | reconstruction_readouts | val | ✅ | ✅ `c_hat=z_c+(1−τ)u` matches train_step:287 |
| `c_attn_entropy*` | attention_entropy | val | ✅ | ✅ per-head (not head-averaged) |

**Conclusion of Walk 2:** the two decision metrics (`c_effective_rank`,
`coarse_vs_copy_ratio`) and `L_sigreg`'s *logged value* are correct and fresh. The single
metric that is **computed/logged misleadingly** is `grad_global_norm` (F1) — and it is
exactly the class of bug described in the brief (a metric that looks broken/flat while the
real signal lives elsewhere, here in `grad_norm`).

---

## Findings — detail + fix (written, not applied)

### F1 — `grad_global_norm` is measured after in-place gradient clipping  ·  MED · pre-existing
`train_step` mutates `.grad` in place via AGC (`train.py:292`) and
`clip_grad_norm_(trainable, 0.5)` (`307`), and does not zero grads at the end. The training
loop then calls `run_diagnostics` (`run_training:556`) → `gradient_health` (`diagnostics.py:259`,
called at `train.py:470`) reads those **already-clipped** grads. With `grad_clip=0.5` and
healthy grads ~2–3, clipping fires on essentially every step, so `grad_global_norm` is
pinned near **0.5** and does **not** reflect the true gradient magnitude. A reader plotting
`grad_global_norm` would conclude grads are tiny/vanishing — the real value is `grad_norm`
(the pre-clip norm `clip_grad_norm_` returns, logged every step).
**Why it matters here:** inv008 health-checks ("did high λ_sigreg blow up the gradient?")
must read `grad_norm`, not `grad_global_norm`. **Fix (choose one):** (a) compute
`gradient_health` **before** clipping (capture grads at the top of `run_diagnostics` won't
work — diagnostics runs on a different/no-backward batch; instead snapshot in `train_step`
pre-clip); or (b) drop `grad_global_norm` from `run_diagnostics` and rely on `grad_norm`; or
(c) rename it `grad_global_norm_postclip` and document it. Recommend **(b)** — `grad_norm`
already covers it.

### F2 — SIGReg runs under bf16 autocast → degraded statistic & gradient  ·  MED · new
`sigreg_loss` is called at `train.py:252`, inside `with autocast_context(...)` (`228`).
Although the function does `z = abstract...float()` (`losses.py:191`), autocast still lowers
the `z @ v` matmul (`202`) to **bf16**, so `y`, `sq`, `pair`, and the `exp(...)` BHEP terms
(`204–212`) all run in bf16 (~3 significant digits). For a regularizer whose job is precise
distributional shaping — and whose **gradient drives B** in inv008 (λ>0) — bf16 is sloppy.
(The existing `covariance_floor` shares the pattern but is never on the gradient, λ_cov=0;
SIGReg is the first reg matmul that actually trains.) **Note:** the *metric*
`c_effective_rank` is unaffected (computed in fp32 outside autocast); this only degrades the
SIGReg **loss/gradient**. **Fix:** force fp32 for the numerical core, e.g. inside
`sigreg_loss` wrap from the matmul onward in
`with torch.autocast(device_type=z.device.type, enabled=False):` (after `z=...float()`),
so the statistic is computed in fp32 regardless of caller context.

### F3 — "λ_sigreg=0 ⇒ byte-identical baseline" is false  ·  MED · new
`sigreg_loss(abstract)` is called **unconditionally** (`train.py:252`), and it consumes the
**global RNG**: `torch.randperm(...)` (`losses.py:196`, since pooled N=2048 > max_rows=512)
and `torch.randn(...)` (`200`) both advance the default generator. The sibling unconditional
calls (`variance_floor`, `covariance_floor`, `slot_diversity_loss`) draw **no** randomness,
so for them the "compute-always, add-when-λ>0 ⇒ byte-identical" contract holds — but for
SIGReg it does **not**: every step's draw shifts the *next* step's `eps_c`/`tau_c`/condition-
dropout, so a λ=0 run on the new code diverges from the pre-SIGReg baseline (e.g.
eager-plant-22). **Practical impact on inv008: ~nil** (all runs use λ>0 and are mutually
consistent; the divergence is statistical, not a result-corrupting bug) — but the **stated
contract is wrong** (code comment `251`, SIGREG_DESIGN §2.2, DESCRIPTION). **Fix (best):**
give SIGReg a **dedicated generator** so it never touches the global stream — e.g.
`torch.Generator(device).manual_seed(base_seed + step)` passed as `generator=`; this both
restores the byte-identical contract AND makes the flow/dropout RNG path identical across
λ values (cleaner ablation). **Alternatively:** gate the call behind `if λ_sigreg>0 or
is_log_step` (still perturbs at λ>0) and correct the docs.

### F4 — gradient can vanish if `c_t` drifts far from unit scale  ·  LOW · new
The BHEP kernel `exp(-½β²(y_j−y_k)²)` saturates to 0 when projected values are large
(scale ≫ 1): then `term1→0`, `term2→0`, and `L_sigreg→term3=1/√3≈0.577` flat with a tiny
gradient. β=1 is tuned for ~unit-scale latents. The variance floor + the bottleneck output
LayerNorm keep `c_t` roughly unit-scale, so this is unlikely — but it's the failure mode to
watch. **Fix/mitigation:** the GUIDE step-0 calibration already prints `L_sigreg`; if it
sits at ~0.577 and flat, the kernel has saturated → lower the effective scale (or raise β).
No code change required now; documented as a launch check.

### F5 — variance_floor and SIGReg pool differently  ·  LOW · doc
`variance_floor` (`losses.py:142`) reshapes to **`(B, n_c·d_c)`** and hinges per-(slot×feature)
std **across the batch**. `sigreg_loss`/`effective_rank` reshape to **`(B·n_c, d_c)`** and act
on the **d_c covariance pooled over batch×slots**. These are genuinely different statistics —
complementary, not opposed (both fight collapse, neither pulls against the other). **Not a
bug**, but the SIGREG_DESIGN §2.3 claim that the var-floor hinge is "inactive once std≥1
where SIGReg lands" is **approximate**: a given (slot,feature) can still trip the hinge even
when the pooled distribution is isotropic. **Fix:** soften that sentence in SIGREG_DESIGN to
"complementary poolings; the hinge is a one-sided net that does not oppose SIGReg," so the
write-up isn't over-precise.

### F6 — unit-test absolute threshold is unverified  ·  LOW · new
`tests/test_sigreg.py::test_sigreg_near_zero_on_isotropic_gaussian` asserts `val < 0.05`.
Couldn't run locally (no torch). For a true-normal sample the seeded BHEP value should be
small (~0.002–0.02 incl. the +1/512 diagonal bias), but I have **not measured it**, and 0.05
may be tight. The robust assertion is the **relative** one (`low_rank > iso + 0.05`).
**Fix:** loosen the absolute bound to ~0.15 (still meaningfully "near zero") and/or print the
value; confirm against the first pod run. The relative test stays as the real check.

### F7 — no gradient-contract test for SIGReg  ·  LOW · new
`models.smoke_test_models` asserts the gradient contracts for var/cov/slot/recon (reaches B,
not EMA/F_c) but not for SIGReg. `test_sigreg.py` checks the gradient is finite/flows to the
**input tensor**, not that it reaches **B and only B**. **Fix:** add to `smoke_test_models`:
`sig = sigreg_loss(bottleneck(detailed)); sig.backward()` then assert grads reach
`bottleneck`, and are `None` on `coarse_flow`/`target_bottleneck`/encoder. Cheap insurance.

### F8 — SIGReg computed every step even on a baseline  ·  TRIVIAL · efficiency
The `O(P·S²)` pairwise tensor (~0.13 GB) is built every step (`252`) even when λ=0 (only the
log value is wanted). Irrelevant to inv008 (all runs λ>0, so it's needed each step), but
wasteful for any future λ=0 baseline. **Fix:** fold into F3's gating (`λ>0 or is_log_step`).

### F9 — `L_recon_pred` is a 0.0 sentinel when option-3 is off  ·  TRIVIAL · doc
With `--lambda-recon-pred 0` (inv008), `L_recon_pred` logs `0.0` every step (`train.py:290`
initialized, only set when λ>0). It is a **loss-contribution sentinel**, not a quality
readout — the future-recon quality metric is the diag `L_recon_chat`. Risk: someone reads
"L_recon_pred=0" as "perfect predicted reconstruction." **Fix:** none needed; noted so the
write-up doesn't misread it.

---

## Verified CORRECT (checked, no action) — so we know these were actually walked

- **BHEP closed form is right** (`losses.py:204–212`): for `Y~N(0,1)`, `term1→1/√(1+2β²)`,
  `term2→2/√(1+2β²)`, `term3=1/√(1+2β²)` ⇒ `T→0`; derived and matches. It's the **V-statistic**
  form (mean includes the diagonal) ⇒ each per-projection value is ≥0, so `.clamp_min(0)`
  is a no-op safety with **no gradient bias**.
- **SIGReg pooling == effective_rank pooling** (`(B·n_c, d_c)`): the loss regularizes exactly
  the distribution the primary metric measures. ✅
- **No stale-metric bug** anywhere: `metrics` is rebuilt each step; diag keys added only at
  diag steps; nothing re-logs a prior step's value. ✅ (the brief's worst case is absent)
- **`coarse_vs_copy_ratio`** math (`diagnostics.py:310–319`): copy velocity `c_t−eps_c` vs
  target `c_plus−eps_c` ⇒ copy_loss = ‖c_t−c_plus‖²; ratio<1 ⇒ model beats "predict no
  change." ✅
- **`c_hat` endpoint** identical in train_step (`287`) and reconstruction_readouts
  (`diagnostics`/`train.py:505`): `z_c+(1−τ)·u`. ✅
- **Decision metrics are fp32** (diagnostics run outside autocast). ✅
- **CLI arch overrides applied before build** (decoder/n_c/horizon/checkpoint). ✅
- **SIGReg gradient target**: `abstract=bottleneck(detailed)`, `detailed` no-grad ⇒ SIGReg
  grad flows into **B only** (not encoder/F_c/decoder/EMA). ✅ (formal test missing — F7)

---

## Recommended fix order (when we act on this)

1. **F1** (drop/rename `grad_global_norm`) — removes a genuinely misleading metric before we
   start reading run health. 1-line.
2. **F2** (fp32 SIGReg) — protect the regularizer gradient that the whole sweep depends on.
3. **F3** (dedicated SIGReg generator) — restores the reproducibility contract + clean
   ablation; also closes F8.
4. **F6 + F7** (test robustness + SIGReg grad-contract) — before trusting `pytest` on the pod.
5. **F4, F5, F9** — doc/launch-check only.

*Re-walk this whole document after applying any fix (per the audit protocol).*
