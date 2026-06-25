# Next steps — fanciful-lake-18

## Status

Run **killed by operator at step 14400** (of 15000) after the trajectory was
unambiguous: stable to the end, rank plateaued at the ceiling, copy gate failing.
Analysis complete. Investigation 006 stays **OPEN** — it answered its secondary
question (cliff immunity) and refuted its primary one (rank enrichment), and it
hands a concrete, evidence-backed lever to the next run.

## What this run settled

- **The Mode-B cliff is solved by recon-into-`B`.** Don't re-test stability; carry it
  forward as a property of any run with `lambda_recon>0`.
- **Option 1 cannot break the rank ceiling at `lambda_recon=0.05`.** Rank is capacity-/
  weight-bound at ~13, not collapse-bound.
- **Prediction is the open problem.** `coarse_vs_copy_ratio` never beat 1.0; `F_c` loses
  to copy-forward. Option 1 has no term that touches this.

---

## The option-3 decision (the live one — and what the tech-lead asked for)

**Pre-registered gate:** promote to option 3 only if run 1 (a) lifts rank without `L_flow`
harm AND (b) shows a large `L_recon_chat − L_recon_cplus` gap.
**Literal reading:** (a) NO (rank flat), (b) NO (gap ~0.007). Gate says *don't*.

**Why we should override the literal gate — grounded in this run's data:**

1. The gate's clause (b) assumed a small `chat−cplus` gap means "`F_c` predicts fine."
   This run **falsifies that assumption**: `F_c` loses to copy by 1.4–2.6×, yet `c_hat`
   reconstructs `e_{t+k}` as well as the true `c_plus`. The present-anchored recon metric
   is **blind to prediction error** — it saturates on static shared content. So a small gap
   is not evidence against option 3; it is evidence the recon signal is in the wrong place.
2. The remaining unsolved gate (copy ratio) is a **prediction** problem, and option 1 has
   **no** prediction term by construction. Option 3 puts the reconstruction gradient *on
   the predicted latent `c_hat` through `F_c`* — directly rewarding `c` geometry that is
   predictable, not merely reconstructable. This is exactly Arbab's VITA-based proposal
   ("train reconstruction on the predicted latent, not the actual latent").
3. The main pre-run risk of option 3 — adding gradient at the unstable `F_c` site — is now
   **much lower**: fanciful shows recon-into-`B` holds `agc_Fc` at ~15 through 14400 steps.

**Honest caveats to carry in:** option 3 is **not** expected to break the rank ceiling on
its own (that looks structural — see open questions). And it changes what `c` optimizes
for, so it could trade some of option 1's stability — watch `agc_Fc_max_ratio` closely.

**Recommendation:** proceed to **option 3 as an *added* branch, not a replacement** — run
the present anchor and the predicted-latent anchor simultaneously (Arbab's "both objectives
at once"). The phasing has served its purpose: option 1 is validated as a stabilizer; turn
on the prediction branch next.

---

## Next run — option 3 (spawn `investigation_006/<new-wandb-name>/`)

Decode the predicted latent and let gradients flow through `F_c`, **keeping** the present
anchor. Code is built for this (generic `D`; `reconstruction_readouts` already computes
`L_recon_chat`) — it is a caller change + a flag, not a rewrite.

Implementation — **DONE** (code in working tree, not yet committed/launched):
- ✅ Through-`F_c` branch added in `train_step`: `recon_pred = reconstruction_loss(decoder(c_hat), target_detailed)`, `c_hat = z_c + (1-tau_c)·u_c_hat` reusing the same `u_c_hat` as `flow_loss` (no extra `F_c` forward), gradient retained through `F_c`.
- ✅ New flag `--lambda-recon-pred` (`cfg.train.lambda_recon_pred`), default 0.0 → byte-identical to fanciful. Reuses decoder `D` and the `recon_warmup` ramp. Logs `L_recon_pred`.
- ✅ **Decision: `B`-gradient routing = LET IT FLOW** (conditioning not detached) — VITA joint training, Arbab's expectation. Documented in `config.py` + `train.py`.
- ✅ `smoke_test_models` inverse contract added: prediction-branch grad must reach `D`, `F_c`, AND `B`, never the EMA bottleneck.
- ⬜ Run smoke test on pod (`python -c "from models import smoke_test_models; smoke_test_models()"`), commit, launch.

Suggested first config (mirror fanciful so the only delta is the new branch):
```bash
python train.py --data ssv2 --steps 15000 --horizon-k 12 --lambda-var 0.5 \
  --lr-coarse-flow 1e-4 --lambda-recon 0.05 --recon-warmup-steps 2000 \
  --lambda-recon-pred 0.05 \
  --log-every 50 --diag-every 500
```
(Run in `tmux` — `tmux new -s recon3_run`, then the command, detach `Ctrl+B` `D`.)

**Success signals:** `coarse_vs_copy_ratio` drops below 1 (and toward 0.70);
`coarse_model_loss` falls while `coarse_copy_loss` does not collapse further; `L_recon_chat`
decouples from `L_recon_cplus` (the gradient now *acts* on it). Rank may or may not move.

**Abort rules** (carry from 005, plus decoder/prediction):
- `L_flow > 1.5` for 200 consecutive steps
- `c_effective_rank` drops > 3 points in 500 steps (Mode-B returning)
- `agc_Fc_max_ratio` median > 200 over any 500-step window (option-3 destabilizing `F_c`)
- `coarse_vs_copy_ratio` rises above the fanciful baseline (~2.6) and keeps climbing

---

## Open questions (may each spawn their own investigation)

1. **Is `>60` even the right rank target for a 128:1 bottleneck?** fanciful sits at ~13/256
   with `c` healthy by every other measure. Re-examine the `PHASE_1.md` §9.2 gate against
   the compression ratio before treating rank as the headline failure.
2. **Is the ~0.6 recon floor a decoder-capacity limit?** `L_recon_present` never improved
   past 0.599. A larger `D` (more blocks/dim) or higher `lambda_recon` ablation would tell
   us whether the floor is `c`'s information limit or `D`'s expressivity limit — relevant to
   whether *any* recon route can lift rank.
3. **Does the late `c_cross_video_cosine` rise (0.24→0.32) continue?** If a longer/option-3
   run pushes it up, that is early Mode-A pressure and `lambda_var` may need a look.
