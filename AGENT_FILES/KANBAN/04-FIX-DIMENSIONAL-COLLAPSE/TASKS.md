# 04 — Tasks: Fix dimensional collapse of `c_t` (coding agent)

> **Read first:**
> - [`DETAILED_UNDERSTAND.md`](DETAILED_UNDERSTAND.md) — the problem, mechanism, success criteria.
> - [`ANALYSIS_AND_DECISIONS.md`](ANALYSIS_AND_DECISIONS.md) — why VICReg-C first, the experiment matrix.
> - [`../README.md`](../README.md) — the agent/human handoff convention (`⏸ PAUSE`, A/H numbering).
>
> **The agent's role here is real code work** (unlike Plan Phase 02, which was
> mostly standby): instrument, gate fixes behind config flags, keep the
> variance-floor-only baseline reproducible, then interpret the human's runs.
>
> **Concurrency note:** the human runs the Fix-1 short job (H4.2) *while* the
> agent implements Fix 2 (A4.3). Do not block on the run to write A4.3.
>
> **Hard rules:** every change is **flag-gated, default off** (so the v0.2
> baseline is one switch away); no encoder change; no Phase 2+ work; all four
> standard guards (no NaN, `grad_skipped`=0, frozen encoder, cross-video
> cosine < 0.5) must keep passing in `smoke`/Stage 0.

---

## Task A4.0 — Instrument: split slot-redundancy from feature-correlation

**Status:** [NOT DONE]

**Description:**
Before any fix, add diagnostics that tell us *which* collapse we have
(`DETAILED_UNDERSTAND.md` §4.3). No training-behavior change.

**Instructions:**

1. In `diagnostics.py`, add functions (pure, returning `dict[str, float]`):
   - `attention_entropy(...)` — mean Shannon entropy of the bottleneck's
     cross-attention weights over the 1024 memory tokens, averaged over the 32
     slots (requires capturing `need_weights=True` from `cross_attn`, or a hook;
     do it without changing the forward's default behavior). Low entropy =
     sharp/peaked attention; high = near-uniform (the saturation symptom).
   - `slot_diversity_rank(abstract)` — effective rank of the **32 slot
     vectors within a single video** (averaged over the batch), i.e. how
     non-redundant the slots are. Distinct from the existing cross-video
     `effective_rank`.
2. Wire both into `run_diagnostics` in `train.py` alongside the existing
   probes so they log to W&B at `diag_every`.
3. Add a tiny `attention_entropy` readout to Stage 0 / `smoke_test_*` so it's
   visible without a full run.

**How to verify:**

- `python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"`
  runs and prints the new keys as floats.
- `python train.py --stage0-only` still passes and now reports attention
  entropy.
- No change to `loss`/`grad_norm` numerics vs before (diagnostics are
  side-effect free).

---

## Task A4.1 — Fix 1: init tweaks, flag-gated

**Status:** [NOT DONE]

**Description:**
The two initialization changes from `ANALYSIS_AND_DECISIONS.md` §4. Hygiene,
not the cure — but cheap and run first.

**Instructions:**

1. Add to `ModelConfig` (config.py), default to current behavior:
   - `bottleneck_zero_init_out_mlp: bool = False`
   - `bottleneck_query_init: str = "small_gaussian"`  (options:
     `"small_gaussian"` = current `randn*0.02`, `"orthogonal"`,
     `"scaled_gaussian"`)
2. In `Bottleneck.__init__` (models.py):
   - When `zero_init_out_mlp`: `nn.init.zeros_(self.out_mlp[-1].weight)` and
     `nn.init.zeros_(self.out_mlp[-1].bias)` (start as `abstract = attended`).
   - Apply the selected `query_init` to `self.queries`
     (`orthogonal` → `nn.init.orthogonal_`; `scaled_gaussian` → larger std,
     e.g. 0.1–0.2, exact value a sweep knob).
3. Leave defaults off so the unflagged build is byte-identical to today.

**How to verify:**

- `python -c "from models import smoke_test_models; smoke_test_models()"`
  passes for each `query_init` option and with `zero_init_out_mlp` on/off.
- With both flags at defaults, the Stage 0 metrics match the pre-change run.
- With flags on, A4.0's `attention_entropy` at step 0 is measurably lower
  (sharper) than with defaults — the empirical check Claude recommended.

---

> **⏸ PAUSE — BLOCKED ON HUMAN (short).**
>
> Hand off to the human to run the **Fix-1 short instrumented job** (H4.2):
> a few hundred steps with the init flags on, reporting at-init attention
> entropy, slot-diversity rank, and `c_effective_rank` trend.
> **Do not block** — proceed to A4.2/A4.3 (implement Fix 2) concurrently.

---

## Task A4.2 — Implement Fix 2a: VICReg covariance term, flag-gated

**Status:** [NOT DONE]

**Description:**
The primary fix (`ANALYSIS_AND_DECISIONS.md` §2): an off-diagonal covariance
penalty on `c_t` to decorrelate the 256 feature dimensions. Completes
VICReg's V (existing variance floor) + C.

**Instructions:**

1. In `losses.py`, add `covariance_floor(abstract: Tensor) -> Tensor`:
   - Flatten to `(-1, D_c)` (pool batch × slots → samples × 256 feature dims),
     center, form the `256×256` covariance.
   - Return the mean of the **squared off-diagonal** entries
     (`(cov.pow(2).sum() - cov.diag().pow(2).sum()) / (d*(d-1))` or the
     standard VICReg `(off_diag**2).sum() / d`). Document the exact convention.
   - Guard `B < 2` → return `0.0` (mirror `variance_floor`).
2. In `config.py` `TrainConfig`: add `lambda_cov: float = 0.0` (default off →
   reproduces the v0.2 baseline exactly). Document that nonzero values are a
   sweep knob (start small, e.g. 0.01–0.1; watch copy-ratio).
3. In `train.py::train_step`, add `cfg.train.lambda_cov * covariance_floor(abstract)`
   to `loss` and log `L_cov` separately.

**How to verify:**

- With `lambda_cov = 0.0`, the total loss and grads are byte-identical to the
  current baseline (the term contributes 0).
- With `lambda_cov > 0`, `smoke_test_models`-style backward shows gradients
  flow to the bottleneck and the loss stays finite.
- `L_cov` appears in logged metrics.

---

## Task A4.3 — (Escalation, optional) Resurrect SIGReg, flag-gated

**Status:** [NOT DONE — escalation only]

**Description:**
Only if Fix 2a underdelivers or the tech lead wants the comparison. The v0.1
implementation exists in git.

**Instructions:**

1. Recover `sigreg(...)` from `git show 5cb8330:losses.py` into `losses.py`
   (random-projection characteristic-function form).
2. Apply to **`c_t` only** (never the frozen `e_t`). Add
   `lambda_sigreg: float = 0.0`, `sigreg_m: int = 1024`, `sigreg_knots: int = 17`
   to `TrainConfig`, default off.
3. Wire into `train_step` behind the flag; log `L_sigreg`.
4. **Mutually exclusive with `lambda_cov` in any single run** (don't confound
   two decorrelation terms). Assert if both are nonzero.

**How to verify:**

- Defaults off → baseline unchanged.
- With the flag on, loss finite, grads flow to bottleneck, `L_sigreg` logged.

---

> **⏸ PAUSE — BLOCKED ON HUMAN (long).**
>
> The main event is the human's larger-data A/B run(s) (H4.3). The human will
> decide the **2-run vs 1-run** version (see `ANALYSIS_AND_DECISIONS.md` §3)
> at launch. Wait for milestone metric reports. Do not hallucinate W&B values.

---

## Task A4.4 — Interpret the A/B run(s)

**Status:** [NOT DONE]

**Description:**
When the human reports run metrics, judge against the success criteria
(`DETAILED_UNDERSTAND.md` §7) — **both** rank and copy-ratio must move the
right way.

**Instructions:**

Assess from the human's reported values:

| Signal | Success reading | Concern |
|---|---|---|
| `c_effective_rank` | rising, > 30 by end (toward 60) | flat near baseline ~5 → the term isn't biting; raise `lambda_cov` or escalate to SIGReg |
| `coarse_vs_copy_ratio` | improving or holding while rank rises | **degrading while rank rises** → Goodhart; the regularizer is inflating the metric without real gain → back off weight |
| `coarse_vs_batch_mean_ratio` | improving | regressing → investigate |
| `c_dead_dim_frac` | < 0.15 | rising → some dims dying despite the term |
| `c_cross_video_cosine` | < 0.5 | drift to 1.0 → directional collapse → abort |
| `slot_diversity_rank` (A4.0) | rose with the fix | if low still → slot redundancy persists (attention saturation) → revisit Fix 1 scale |
| guards | no NaN, `grad_skipped`=0, encoder unchanged | any breach → stop |

Then compare A vs baseline (data effect) and B vs A (regularizer effect) if
the 2-run version was used.

**How to verify:**

- A written verdict per run: "fix succeeded / partial / failed", with the
  rank-and-copy-ratio joint reading, sent to the human.

---

## Task A4.5 — Decide escalation and close out

**Status:** [NOT DONE]

**Description:**
Based on A4.4, pick the next move and close the folder.

**Instructions:**

1. Recommend one of:
   - **Success** → lock the winning flags as the new Phase 1 default; fold
     into a fresh `02`-style launch; proceed toward the Phase 1 acceptance gate.
   - **Partial / no rank gain** → escalate to SIGReg (A4.3) or raise
     `lambda_cov`; re-run.
   - **Rank up but copy-ratio down** → reduce weight; if irreconcilable,
     reconsider whether the task is too easy (Fix 4 / tubelet-horizon) with
     the tech lead.
2. Update both this file and `HUMAN_TASKS.md` statuses to `[DONE]`.
3. Note whether the supervisor needs to sign off on keeping a covariance term
   (it reverses the v0.2 "no covariance loss initially" directive — see
   `DETAILED_UNDERSTAND.md` §8).

**How to verify:**

- Recommendation delivered; statuses closed; supervisor sign-off flagged.
