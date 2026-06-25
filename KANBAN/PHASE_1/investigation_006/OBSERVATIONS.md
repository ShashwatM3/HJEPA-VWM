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
