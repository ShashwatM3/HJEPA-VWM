# Observations — run 056 `ae_latent_stack_whiten_abs_recon_geom`

**W&B run:** [`tl5dh73c`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/tl5dh73c)

**State:** finished, 15,000/15,000 steps

**Mode:** present-only; `F_c`, future prediction, and copy/batch-mean gates are inactive

**Verdict:** **Strong present representation**, with a measurable honesty/decodability tax from
the full geometry bundle

## Reading Cycle B

| Q | Question | Result | W&B evidence |
|---|---|---|---|
| Q1 | Trained alive and in the intended mode? | PASS | 300 history rows, 30 diagnostic rows, final step 14,950; zero skipped steps, NaN flags, or instability warnings; `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0`, `whiten_active=1`, and `recon_target_residual=0` throughout. Maximum logged pre-global-clip `grad_norm` was 6.24 during warmup and fell to 0.179 at the final row; logged B/D AGC event counts stayed zero. |
| Q2 | Is `c_t` alive and video-specific? | PASS | Final `c_std_mean=1.0276`, dead fraction 0, and `c_cross_video_cosine=0.0168`. The code is spread and different videos are nearly orthogonal on the SSv2 fixed batch. |
| Q3 | Is the latent rich rather than mechanically rank-31? | PASS | Rank climbed from the mechanical slot-identity value 30.99 to `201.58/256`; centered slot rank held at `29.87/32`. The learned rank is far above what 32 fixed slot identities can explain. |
| Q4 | Is reconstruction using the supplied code? | PASS, with a tax | `L_recon_present` fell `1.00375 -> 0.73076`; shuffled-code loss ended `0.93438`, so the gap was `0.20362`. About 74.6% of the learned improvement required the correct code. |
| Q5 | Do geometry and reconstruction cooperate? | PASS, not optimal | Rank, spread, video separation, and slot diversity all remained healthy while reconstruction learned. Compared with run 055, geometry improved dramatically, but raw reconstruction and conditioned share worsened, showing a decodability/honesty cost. |
| Q6 | Verdict | Strong present representation | First run in this architecture family to combine genuinely high-rank, spread geometry with clearly code-conditioned reconstruction. It is present-side evidence only, not a prediction success. |

## Trajectory landmarks

| Step | Effective rank | Mean std | Cross-video cosine | Centered slot rank | Present recon | Shuffled recon | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.99* | 0.000 | 1.000 | 30.99* | 1.00375 | 1.00375 | 0.00000 |
| 1,000 | 75.23 | 0.933 | 0.0707 | 29.62 | 0.89181 | 0.94137 | 0.04956 |
| 2,000 | 121.67 | 1.001 | 0.0327 | 29.58 | 0.82642 | 0.93067 | 0.10425 |
| 5,000 | 159.03 | 0.987 | 0.0777 | 28.56 | 0.76514 | 0.92949 | 0.16435 |
| 10,000 | 188.42 | 1.017 | 0.0307 | 29.54 | 0.73830 | 0.93670 | 0.19840 |
| 14,500 | 201.58 | 1.028 | 0.0168 | 29.87 | 0.73076 | 0.93438 | 0.20362 |

`*` Step-zero rank is the known fixed-slot-identity scaffold, not learned sample information.

## What the immediate successor established

Run 057 (`cdvp6hou`) was documented and launched directly after this run with exactly one
scientific removal: `lambda_sigreg: 5 -> 0`; covariance `0.01`, variance `0.5`, whitening,
absolute target, architecture, seed, and schedule stayed fixed.

| Final diagnostic | Run 056: cov + var + SIGReg | Run 057: cov + var | Consequence |
|---|---:|---:|---|
| Effective rank | 201.58 | **208.22** | SIGReg was not required for rank. |
| Centered slot rank | 29.87 | **30.71** | Slots remained differentiated without SIGReg. |
| Mean std | 1.028 | 1.117 | The one-sided variance floor allowed a harmless overshoot. |
| Cross-video cosine | **0.0168** | 0.0585 | SIGReg bought a small extra separation benefit, but both are healthy. |
| Present reconstruction | 0.73076 | **0.71285** | Removing SIGReg improved decodability. |
| Video gap | 0.20362 | **0.22681** | Removing SIGReg improved correct-code dependence. |
| Conditioned share | 74.6% | **78.0%** | SIGReg imposed a small honesty tax. |
| Attention mean / minimum | 0.469 / 0.006 | **0.721 / 0.278** | SIGReg caused the near-delta head specialization; broad reads were sufficient. |

The successor therefore refines, rather than reverses, this run's result: explicit anti-collapse
pressure is required, covariance is the working rank lever, the variance floor protects amplitude,
and full-distribution SIGReg is unnecessary at this operating point.

## Evidence surfaces checked

- W&B config: exact run-055 base plus `lambda_var=0.5`, `lambda_cov=0.01`, and
  `lambda_sigreg=5`, with the 2,000-step SIGReg ramp.
- Full unsampled metric history: all 300 training rows and all 30 diagnostic checkpoints.
- Console log: 300 step records from 0 through 14,950 with no traceback or numerical failure.
- System stream: A100 telemetry throughout the run; no resource termination signature.
- W&B files/artifacts: config, summary, metadata, requirements, console log, history parquet, and
  events parquet. This older run did not log a checkpoint/provenance artifact to W&B.
- Full interpretation: [`ANALYSIS.md`](ANALYSIS.md).

## Reading constraints

- This is present-only. It neither passes nor fails the Phase 1 copy or batch-mean prediction
  gates, and `F_c` learned nothing scientifically interpretable here.
- Whitened-space reconstruction values must not be compared numerically with raw-feature runs.
- Raw slot rank and step-zero pooled rank are mechanically inflated by fixed slot identities;
  the learned feature rank, across-video spread, and shuffled-code probe are the relevant checks.
