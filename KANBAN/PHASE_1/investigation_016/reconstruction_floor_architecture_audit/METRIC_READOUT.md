# Metric readout — reconstruction-floor architecture audit

> Fetched with `python run_history.py --run <id> --report` and unsampled W&B
> histories. Completed runs were read through all 300 logged rows / 30 diagnostic
> checkpoints. The run-060 block below is a preserved 2026-07-16 live snapshot through step 5,450;
> the final reconciliation is recorded immediately after it.

## Run identities

| Role | Dataset | W&B ID | State at read | Last logged step | Git commit |
|---|---|---|---|---:|---|
| SSv2 control (057) | `ssv2` | [`cdvp6hou`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/cdvp6hou) | finished | 14,950 | `5ea4421ff7aa034894dc07ab6270ae766a00d83a` |
| EGO4D target (058) | `ego4d` | [`mvbx96nv`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/mvbx96nv) | finished | 14,950 | `21d2aa8a97e8f60536a681bc5a4324d1a2c28838` |
| EGO4D weight arm (060), historical snapshot | `ego4d` | [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g) | running at snapshot | 5,450 | `a27cd84dd67783f7ee8e68bb68e7c1a38a309534` |
| EGO4D weight arm (060), final reconciliation | `ego4d` | [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g) | finished | 14,950 | `a27cd84dd67783f7ee8e68bb68e7c1a38a309534` |

## Config highlights

All three runs report:

- `present_recon_only=true`, so `L_flow=0`, `prediction_active=0`, and F_c is not part
  of the optimized task;
- absolute targets: `recon_residual_target=false`;
- cosine reconstruction;
- fixed feature whitening with `eps=1e-4`;
- `lambda_var=0.5`, `lambda_cov=0.01`, `lambda_sigreg=0`, `lambda_slot=0`;
- batch 64, seed 42, 15,000 steps, 1,500-step LR warmup, 2,000-step reconstruction
  ramp;
- bottleneck `n_c=32`, `d_c=256`, mixer width 256, three latent blocks;
- decoder width 512 and four blocks;
- B/D peak LR `1e-4`, Adam betas `(0.9, 0.95)`, weight decay `0.05`, and global
  gradient clip `0.5`.

The run-defining config differences are:

| Field | Run 057 | Run 058 | Run 060 |
|---|---:|---:|---:|
| dataset | SSv2 | EGO4D | EGO4D |
| whitening path | SSv2 stats | EGO4D stats | strict EGO4D stats artifact |
| `lambda_recon` | 0.05 | 0.05 | 1.00 |

Run 057 to run 058 retained the same `models.py` and `losses.py`; the intervening
changes added EGO4D selection/support and related configuration/training plumbing. Run
060 is on the newer deterministic-data, pinned-encoder, strict-artifact/provenance
pipeline.

## Completed-run terminal comparison

The diagnostic values below are the final step-14,500 fixed-batch measurements. The
training loss is the final step-14,950 random-batch measurement.

| Metric | Run 057 SSv2 | Run 058 EGO4D |
|---|---:|---:|
| train `L_recon` @14,950 | 0.707859 | 0.696892 |
| fixed-batch `L_recon_present` @14,500 | 0.712850 | 0.678246 |
| `L_recon_shuffled_c` | 0.939664 | 0.696221 |
| recorded `L_recon_video_gap` | 0.226814 | 0.017975 |
| `c_effective_rank` | 208.215 | 52.910 |
| `c_cross_video_cosine` | 0.058538 | 0.863109 |
| `c_std_mean` | 1.11725 | 0.419427 |
| centered slot rank | 30.71 | 30.67 |
| skipped optimizer steps | 0 | 0 |
| NaN-gradient rows | 0 | 0 |

Late random-training-batch windows:

| Window | Run 057 mean +/- population SD | Run 058 mean +/- population SD |
|---|---:|---:|
| steps 10,000-14,950 | 0.707895 +/- 0.003693 | 0.697178 +/- 0.004012 |
| steps 12,500-14,950 | 0.706951 +/- 0.003343 | 0.696103 +/- 0.003874 |
| steps 14,000-14,950 | 0.706427 +/- 0.003515 | 0.694654 +/- 0.003552 |

Late fixed-batch windows:

| Window | Run 057 mean / last | Run 058 mean / last |
|---|---:|---:|
| steps 10,000-14,500 | 0.714754 / 0.712850 | 0.679123 / 0.678246 |
| steps 12,500-14,500 | 0.713288 / 0.712850 | 0.678362 / 0.678246 |

## Present-only reading cycle

### Run 057 — `cdvp6hou`

| Q | Question | Read | Evidence |
|---|---|---|---|
| Q1 | Training alive? | pass | Finished 15,000 steps; no skips/NaNs; final `grad_norm=0.0759`. |
| Q2 | Code alive/sample-specific? | pass on recorded SSv2 batch | std 1.117, dead fraction 0, pairwise cosine 0.059. |
| Q3 | Rich latent? | pass | rank 208.2/256; centered slot rank 30.71/32. |
| Q5-prime | Honest present reconstruction? | pass | present 0.7129 vs shuffled 0.9397; gap 0.2268. |
| Verdict | | **Strong present representation** | Finished, high-rank, spread, sample-specific, and decodable. |

### Run 058 — `mvbx96nv`

| Q | Question | Read | Evidence |
|---|---|---|---|
| Q1 | Training alive? | pass | Finished 15,000 steps; no skips/NaNs; stable gradients. |
| Q2 | Code alive/sample-specific? | weak on recorded batch | std 0.419; recorded pairwise cosine 0.863. |
| Q3 | Rich latent? | weak on recorded batch | rank 52.9; centered slot rank 30.67. |
| Q5-prime | Honest present reconstruction? | unresolved globally | present 0.6782 vs rolled-code 0.6962; gap 0.018 on a single-source batch. |
| Verdict | | **No valid global label from this batch** | Raw reconstruction learned; the recorded batch measures within-source adjacent chunks. |

This replaces neither the historical record nor its original label. It records the
later-discovered batch-identity limitation that changes what the metrics can support.

## Run-058 diagnostic trajectory

| Step | Present | Rolled code | Gap | Rank | Pairwise cosine | Std |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0133 | 1.0133 | 0.0000 | 30.99 | 1.000 | ~0 |
| 500 | 0.9825 | 0.9829 | 0.0004 | 40.56 | 0.596 | 0.573 |
| 1,000 | 0.9082 | 0.9115 | 0.0033 | 65.38 | 0.369 | 0.746 |
| 2,000 | 0.8193 | 0.8264 | 0.0071 | 80.73 | 0.515 | 0.705 |
| 5,000 | 0.7095 | 0.7225 | 0.0131 | 63.10 | 0.793 | 0.514 |
| 7,500 | 0.6872 | 0.7024 | 0.0152 | 51.01 | 0.871 | 0.408 |
| 10,000 | 0.6820 | 0.6997 | 0.0177 | 57.52 | 0.833 | 0.464 |
| 12,500 | 0.6789 | 0.6966 | 0.0177 | 53.04 | 0.862 | 0.421 |
| 14,500 | 0.6782 | 0.6962 | 0.0180 | 52.91 | 0.863 | 0.419 |

## Historical live run-060 snapshot — no final verdict at that time

| Step | Present | Rolled code | Gap | Rank | Pairwise cosine | Std |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0163 | 1.0163 | 0.0000 | 31.00 | 1.000 | ~0 |
| 1,000 | 0.8654 | 0.8729 | 0.0075 | 43.25 | 0.895 | 0.286 |
| 2,000 | 0.7557 | 0.7704 | 0.0148 | 43.12 | 0.928 | 0.268 |
| 3,000 | 0.7180 | 0.7359 | 0.0179 | 81.55 | 0.586 | 0.682 |
| 3,500 | 0.7024 | 0.7202 | 0.0178 | 112.73 | 0.404 | 0.846 |
| 4,000 | 0.6995 | 0.7191 | 0.0196 | 108.89 | 0.457 | 0.828 |
| 4,500 | 0.6957 | 0.7164 | 0.0206 | 94.50 | 0.533 | 0.767 |
| 5,000 | 0.6899 | 0.7112 | 0.0212 | 83.26 | 0.598 | 0.716 |

At snapshot step 5,450, train `L_recon=0.69212`, `grad_norm=0.1565`, LR multiplier
`0.8032`, skipped steps 0, B AGC events 0, and D AGC events 0. Of 110 logged rows,
33 had a pre-global-clip norm above 0.5.

## Final run-060 addendum

Run 060 later finished all 15,000 scheduled updates. W&B contains 300 training rows through step
14,950 and 30 diagnostics through step 14,500, with zero skipped steps, NaN-gradient rows, or
instability warnings. The final fixed-batch read is:

| Metric | Final value |
|---|---:|
| `L_recon_present` | 0.670912 |
| `L_recon_shuffled_c` | 0.697356 |
| exact-chunk gap | 0.026444 |
| `c_effective_rank` | 84.363 |
| `c_std_mean` | 0.8004 |
| within-source pair cosine | 0.4790 |

The batch contains one source UID, so the gap is exact-chunk/within-source evidence. The completed
Reading Cycle B and causal limitations are in
[`../run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md`](../run_060_ae_latent_stack_whiten_abs_recon_cov_var_ego4d_recon1/ANALYSIS.md).

## EGO4D validation-batch identity

The strict provenance envelope for run 060 records the 16 validation sample IDs. They
are:

```text
validation/01cab463-9a16-4817-84a4-a00ef5b7bf39_00000.mp4
...
validation/01cab463-9a16-4817-84a4-a00ef5b7bf39_00015.mp4
```

All share source UID `01cab463-9a16-4817-84a4-a00ef5b7bf39`. Current `data.py` sorts
paths and uses `range(len(validation))`; `provenance.py` records the same first-16 rule.
At the run-058 commit, `data.py` also sorted paths, the validation DataLoader used
`shuffle=false`, and `run_training` consumed the first batch. Thus the same ordering
property applies to the historical EGO4D run.

## Current EGO4D whitening artifact facts

Artifact payload fingerprint:
`fd00b842eebeaa2f5a83a47b82275b39182a58c91131793fd48e3502fe73d0b6`.

| Field | Value |
|---|---:|
| source clips | 12,800 |
| token rows | 13,107,200 |
| channel dimension | 1,024 |
| raw covariance eigenvalue sum | 6,895.9735 |
| raw channel effective rank | 230.4794 |
| raw eigenvalue min / median / max | `2.63e-13` / `1.6567` / `569.9024` |
| eigenvalues below `eps=1e-4` | 1 |
| post-whitening covariance trace | 1,022.8863 |
| post-whitening effective rank | 1,022.99999 |
| post-whitening min / max eigenvalue | `2.63e-9` / `0.9999998` |
| whitening gain min / max | `0.04189` / `100.0` |

For the post-whitening eigenvalues, the sum of the largest 256 divided by the total is
`0.25026968`; its square root is `0.50026961`.

## Schedule and clipping facts

| Step | LR multiplier | Cumulative LR-multiplier area already consumed |
|---:|---:|---:|
| 1,500 | 1.0000 | ~10.0% |
| 5,000 | 0.8431 | 54.17% |
| 7,500 | 0.5868 | 78.21% |
| 10,000 | 0.3020 | 92.97% |
| 12,500 | 0.0823 | 99.08% |
| 14,000 | 0.0135 | 99.94% |
| 14,500 | 0.00338 | >99.98% |

In run 058, the global 0.5 norm clip condition held on 90% of logged rows before step
1,500, 75% before step 2,000, and 30% before step 5,000. It held on 10% of all 300
logged rows. B AGC and D AGC never activated in the sampled run-058 history, and no
step was skipped.
