# Observations — investigation 006

No run data yet. This section records the **pre-run analysis** that motivated the
mechanism; metrics and interpretation are appended as dated sections once runs land.

---

## 2026-06-25 — Pre-run analysis (W&B MCP, all 17 runs)

**Two distinct failure modes exist in the run history — they were being conflated.**

- **Mode A — representational collapse.** `copper-sky-12`, `olive-terrain-11`,
  `skilled-waterfall-10` (ran `lambda_var=0.1` + slot/cov): `c_attn_entropy → 0.9999`
  (uniform attention), `c_std_mean → 0.46–0.60`, `c_cross_video_cosine → 0.65–0.75`.
  **Fixed** by `lambda_var=0.5` — `cerulean/elated/drawn/royal` hold std ≈ 1.0,
  cosine ≈ 0.16–0.26 stably.
- **Mode B — optimization-cliff dimensional collapse.** `royal-cherry-17`: healthy
  to step 8500 (rank 13.9, std 1.04, cosine 0.26, attn 0.92 — NOT uniform), then at
  8600 `L_flow` 0.26→3.1, `agc_Fc_max_ratio`→1284; over 2500 steps rank 13.9→**5.8**,
  slot rank 19.3→5.7, **while std stays ~0.94 and cosine only drifts to 0.35–0.41**.
  Copy baseline stayed easy (0.14) throughout — `c` still distinguished videos;
  **F_c's forward prediction broke** and dragged the bottleneck down.

**Implications for this investigation:**

1. The reconstruction anchor's classic justification ("identical-`c` cheat lowers
   `L_flow`") targets **Mode A**, which is already controlled. The live failure is
   **Mode B**, where `L_flow` goes *up*, not down — no rewarded cheat.
2. The honest, data-backed reason to add reconstruction is the **rank ceiling**
   (~13.9/256 even when healthy) — an information-richness gap, not collapse per se.
3. **Mode B is an `F_c` optimization event.** Option 1 (gradient into B only) does
   not route through `F_c`, so it can only help Mode B *indirectly* via richer/less-sharp
   `c`. Routing recon *through* `F_c` (option 3) adds gradient load at the empirically
   unstable site — deferred until option 1 shows rank lift without `L_flow` harm.
4. The brief's stop-grad rule #2 ("`c_hat` detached before `F_e`; stops `c` becoming a
   texture carrier") was a **pre-experiment prior, not a result**. On mechanism the
   texture-carrier risk is real but **capacity-bounded** (`c` is 8,192 numbers vs `e`'s
   ~1M, ~128:1) and self-regulating via `L_flow` — a tunable risk, not a blocker.

**Signals to read once runs land:**

- `c_effective_rank` — does it climb past ~13 toward the >60 gate?
- The 8000–9000 window — does the cliff soften / shift with richer `c`?
- `coarse_vs_copy_ratio` and `L_flow` — recon must NOT degrade prediction.
- `L_recon_chat` − `L_recon_cplus` gap — the data-driven warrant for option 3.

---

## 2026-06-25 — fanciful-lake-18 (first active run, λ_recon=0.05) — results

Full analysis in [`fanciful-lake-18/OBSERVATIONS.md`](fanciful-lake-18/OBSERVATIONS.md).
Run reached step 14400 (killed by operator), full SSv2, royal-cherry regime + recon.
Reading the signals above against the data:

| Signal | Pre-run question | What the run showed |
|---|---|---|
| `c_effective_rank` | climb past ~13? | **No** — plateaued 13.1–13.3, same ceiling. Primary hypothesis refuted. |
| 8000–9000 cliff | soften? | **Removed entirely.** `agc_Fc` peaked 17.9 (royal: 1284); rank held; 0 skips to 14400. |
| `coarse_vs_copy_ratio` / `L_flow` | no regression? | `L_flow` healthy (~0.38). Copy gate **persistently failed** (floor 1.43, end 2.59) — first sustained honest read. |
| `L_recon_chat − cplus` | option-3 warrant? | Gap ~0.007 (tiny) — but this **refutes the gate's premise**, see below. |

**Synthesis — option 1 is a stability win and a rank-enrichment null.** The anchor's real
effect is regularizing `B` so `F_c` never enters the Mode-B blow-up (confirming the
*secondary* hypothesis, not the primary one). It does **not** enrich rank and has **no**
prediction term, so the copy gate stays failed.

**Belief update on the option-3 gate.** The pre-run gate (Signals, above) assumed a small
`chat−cplus` gap means "`F_c` predicts fine → option 3 unwarranted." fanciful **falsifies
that**: `F_c` loses to copy by 1.4–2.6× *while* `c_hat` reconstructs `e_{t+k}` as well as
`c_plus`. Conclusion: **present-anchored recon is blind to prediction error** (it saturates
on static shared content at the ~0.6 floor). So the small gap is evidence the recon signal
is in the *wrong place*, not evidence against routing it through `F_c`. The data-backed next
lever is therefore **option 3 / the predicted-latent anchor through `F_c`** (= the tech-lead's
VITA-based proposal), run *alongside* the present anchor. Carry the caveat: option 3 is not
expected to break the rank ceiling (looks structural at 128:1 compression — open question).

---

## 2026-06-25 — easy-blaze-19 (option 3, λ_recon_pred=0.05) — NEGATIVE RESULT

Full analysis: [`easy-blaze-19/OBSERVATIONS.md`](easy-blaze-19/OBSERVATIONS.md). Clean A/B vs
fanciful (same seed/config, only `lambda_recon_pred` 0 → 0.05); completed full 15k.

**Option 3 did not work.** Routing reconstruction through `F_c` on the predicted latent:
- **No prediction gain** — `coarse_vs_copy_ratio` tracks fanciful, ends slightly worse (2.95 vs 2.59); `L_recon_chat` did not drop; `L_recon_pred` pinned at ~0.64 all run.
- **Mild representation harm** — rank 12.6 vs 13.1, cross-video cosine 0.41 vs 0.32, std 0.95 vs 1.01 (gentle Mode-A drift, late-accelerating).
- **Stability fine** — no cliff, `agc_Fc` 9–18, 0 skips. The option-3 risk did not materialize.

**Why — the capacity floor (decisive).** In *both* runs `L_recon_present ≈ cplus ≈ chat ≈ pred
≈ 0.60`. `c` (8,192 numbers) can only rebuild ~40% of `e`; the rest is unreachable at 128:1.
So a good and a bad prediction reconstruct identically → the recon objective is **blind to
prediction quality at every τ**, and `F_c` gets no corrective gradient. Training-through-`F_c`
cannot beat a structural floor.

**Belief correction:** the prior section concluded option 3 was the data-backed lever to fix
prediction. easy-blaze **refutes that** — it's the right *target* but the reconstruction
*channel* is too low-capacity to carry the signal. Also: **cheap-vs-clean `c_hat` is moot**
(both hit the floor).

**Revised open question (the live one):** is the 0.60 floor **weight-bound or capacity-bound**?
A `lambda_recon=0.2` ablation answers it cheaply (does `L_recon_present` drop below ~0.55?). If
weight-bound → bigger λ / bigger `c`, then re-test option 3. If capacity-bound → reconstruction
is the wrong lever for prediction; pivot to the **task/horizon** (copy is strong because `c`
barely moves over horizon-12: `‖Δc‖/‖c‖` ~0.38 and falling). See
[`easy-blaze-19/NEXT_STEPS.md`](easy-blaze-19/NEXT_STEPS.md).
