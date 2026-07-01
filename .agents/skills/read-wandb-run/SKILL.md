---
name: read-wandb-run
description: >-
  Read and interpret HJEPA-VWM W&B training runs using the project's eight-question
  reading cycle (stability, collapse, rank, static-c, copy gate, conditioning, recon
  blindness, verdict). Use when the user asks to analyze, compare, diagnose, summarize,
  or review experiment runs, metrics panels, or run history in this repository — one
  run or many.
---

# Read HJEPA-VWM W&B runs

## Mandatory workflow

**Always** use the reading cycle in
[`KANBAN/README_for_reading_experiments.md`](../../KANBAN/README_for_reading_experiments.md).
Do not improvise a generic ML metric tour. Do not list every logged metric.

1. **Read** `KANBAN/README_for_reading_experiments.md` (full Q1–Q8 + Q5′).
2. **Fetch run data** (one or more runs):
   - W&B MCP / UI: project `hjepa-vwm`
   - CLI: `python run_history.py --run <id> --report`
   - Or: `wandb-primary` skill for API access
3. **Read run config** before Q1: `present_recon_only`, `predict_residual`, `lambda_sigreg`,
   `lambda_recon`, `horizon_k`, step count, dataset (`ssv2` vs `ssv2_tiny`).
4. **Apply Q1 → Q8 in order** (or Q1 → Q5′ → Q8 for present-recon-only). Max **four panels**
   per question — only those named in the README.
5. **Metric semantics** if confused: `AGENT_FILES/KNOWLEDGE/PROBLEMS_METRICS_AND_EXPERIMENTS.md`
   (coarse / copy section: copy loss is **not** a training loss).

## Single run vs multiple runs

| Task | Procedure |
|---|---|
| **One run** | Q1–Q8 sequentially → Q8 verdict label → 2–3 sentence summary |
| **Multiple runs** | Full cycle **per run** → Q8 label each → comparison table on gates + Q4 pattern |
| **vs baseline** | Same cycle on both; diff config keys explicitly; never compare copy ratio across incompatible modes (full-latent vs residual) without noting `predict_residual` |

Do not average metrics across runs before labeling each run.

## Output template

Copy and fill for every analysis request:

```markdown
## Run: <name> (<id>)

**Config highlights:** <present_recon_only, predict_residual, key lambdas, steps>

| Q | Question | Pass? | Evidence (metrics @ step) |
|---|----------|-------|---------------------------|
| Q1 | Training alive? | | grad_skipped, grad_has_nan, grad_norm |
| Q2 | c_t alive / video-specific? | | c_std_mean, c_dead_dim_frac, c_cross_video_cosine |
| Q3 | Rich latent (not rank-13)? | | c_effective_rank, c_plus_effective_rank |
| Q4 | Temporal dynamics (not static c)? | | coarse_copy_loss trend, ratio, rank |
| Q5 | F_c beats copy? | | coarse_vs_copy_ratio, model/copy losses |
| Q6 | F_c uses c_t? | | coarse_vs_batch_mean_ratio |
| Q7 | Recon honest? | | L_recon_present, L_recon_cplus, L_recon_chat |
| Q8 | **Verdict** | | <label from README> |

**Summary:** <what happened, which failure mode if any, one concrete next step>
```

For **multi-run**, add:

```markdown
## Comparison

| Run | Q8 verdict | copy ratio | batch-mean ratio | Q4 pattern |
|-----|------------|------------|------------------|------------|
| ... | ... | ... | ... | static-c / no-predictor / passing |
```

## Hard rules

- **`coarse_copy_loss` is diagnostic only** — never describe it as optimized or “should go down.”
- **Passing representation ≠ passing run** — Q5 copy ratio ≤ 0.70 is required for full prediction runs.
- **`L_flow` down alone is not success** — always report Q5.
- **inv010 lesson:** rank > 60 with ratio ~ 1.06 = “healthy rep, no predictor”, not amazing.
- **Present-recon-only:** skip Q5–Q6; use Q5′; do not cite copy-ratio gates.
- Prefer **end-of-run** or **last stable window** diag steps; note if run died early (Q1).

## When to escalate

- Q1 fail → cite step of first skip/NaN; recommend checkpoint before spike if resuming.
- Q4 static-`c` + Q5 ratio ~1 → cite SIGReg/recon/horizon from config; do not recommend “more L_flow”.
- Q7 blind + Q5 fail → reconstruction channel not fixing prediction; say so explicitly.
