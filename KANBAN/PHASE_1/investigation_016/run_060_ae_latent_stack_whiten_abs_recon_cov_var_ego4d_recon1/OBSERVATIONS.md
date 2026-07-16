# Observations — run 060

**W&B run:** [`2423b84g`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/2423b84g)

**State:** finished, 15,000/15,000 steps

**Mode:** present-only; prediction and the copy/batch-mean gates are inactive

**Verdict:** **Geometrically healthy on the recorded batch, weakly conditioned on exact chunk,
source-confounded, and not prediction-ready**

## Reading Cycle B

| Q | Question | Result | W&B evidence |
|---|---|---|---|
| Q1 | Trained alive and in the intended mode? | PASS | 300 training rows and 30 diagnostics through steps 14,950/14,500; zero skipped steps, NaN flags, or instability warnings; `present_recon_only=1`, `prediction_active=0`, `L_flow=0`, `L_recon_pred=0`, `whiten_active=1`, `recon_target_residual=0`. Maximum logged pre-global-clip `grad_norm` was 5.02; final was 0.116. |
| Q2 | Is `c_t` alive and sample-specific? | CONDITIONAL PASS | Final std `0.800`, dead fraction 0, and recorded pair cosine `0.479`. Those pass the historical numerical gates on this batch, but all 16 samples are adjacent chunks from one source UID, so the result is within-source rather than cross-source. |
| Q3 | Is the latent rich rather than mechanically rank-31? | PASS WITH CAVEAT | Final effective rank `84.36`, late median `85.06`, centered slot rank `30.81`. This is above the rank-60 gate, although pooled rank still mixes slot identity with sample variation. |
| Q4 | Is reconstruction learning content from the supplied code? | WEAK | Present loss fell `1.01627 -> 0.67091`, but rolled-code loss also reached `0.69736`. Gap `0.02644`; only 7.66% of the learned improvement depends on the exact chunk code. |
| Q5 | Do geometry and content cooperate? | MIXED | Geometry finishes healthy while reconstruction improves, but the honesty probe stays weak and the source-confounded batch cannot establish global conditioning. |
| Q6 | Verdict | Stable geometry, weak exact-chunk conditioning | Valid present-only optimization result; not a predictor result and not a globally validated representation win. |

## Metric trajectory

| Step | Effective rank | Mean std | Recorded pair cosine | Present recon | Rolled-code recon | Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.00* | 0.000 | 1.000 | 1.01627 | 1.01627 | 0.00000 |
| 1,000 | 43.25 | 0.286 | 0.895 | 0.86539 | 0.87290 | 0.00751 |
| 2,000 | 43.12 | 0.268 | 0.928 | 0.75565 | 0.77042 | 0.01477 |
| 3,500 | 112.73 | 0.846 | 0.404 | 0.70239 | 0.72023 | 0.01784 |
| 7,500 | 60.32 | 0.577 | 0.729 | 0.67744 | 0.70128 | 0.02384 |
| 10,000 | 110.36 | 0.923 | 0.345 | 0.67402 | 0.69948 | 0.02546 |
| 14,500 | 84.36 | 0.800 | 0.479 | 0.67091 | 0.69736 | 0.02644 |

`*` The fixed slot identities mechanically produce rank near 31 at initialization despite zero
cross-sample spread.

## Comparison with the preceding run

Run 058 (`mvbx96nv`) is the preceding EGO4D science run and the intended weight-0.05 control.

| Final diagnostic | Run 058: weight 0.05 | Run 060: weight 1.0 | Read |
|---|---:|---:|---|
| Present reconstruction | 0.67825 | **0.67091** | Only 0.00733 absolute / about 1.08% relative improvement. |
| Rolled-code reconstruction | 0.69622 | 0.69736 | The generic/within-source channel did not disappear. |
| Gap | 0.01797 | 0.02644 | Larger, but still small and source-confounded. |
| Exact-chunk conditioned share | 5.36% | 7.66% | Only +2.30 percentage points. |
| Effective rank | 52.91 | **84.36** | Geometry is observationally better. |
| Mean std | 0.419 | **0.800** | Geometry is observationally better. |
| Recorded pair cosine | 0.863 | **0.479** | Within-source sample separation is better. |

The apparent geometry improvement cannot be assigned to `lambda_recon`: the two runs used
different commits and materially different initialization, data-order, encoder, artifact, and
provenance contracts. A same-commit paired control is required for causality.

## Measurement-contract correction

The W&B provenance artifact records validation chunks `..._00000.mp4` through
`..._00015.mp4`, all from source UID `01cab463-9a16-4817-84a4-a00ef5b7bf39`. Therefore:

- `c_cross_video_cosine` is a within-source adjacent-chunk cosine;
- `torch.roll(c, 1)` substitutes another chunk from the same recording;
- 7.66% is an exact-chunk conditioned share, not a global video-conditioned share;
- global template collapse and global cross-source specificity remain unanswered.

## Evidence surfaces checked

- Resolved W&B config before metric interpretation: EGO4D, absolute whitened target,
  `lambda_recon=1`, `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg off, 32x256 latent, decoder 512x4.
- Full unsampled history: all 300 training rows and all 30 diagnostic checkpoints.
- Console log: 300 records with no traceback, numerical failure, or skipped update.
- System telemetry: A100 run, finite resource history, no resource-failure signature.
- W&B artifacts: run provenance, exact 4.2 MB whitening payload, history parquet, and events
  parquet; provenance records the dataset/encoder/whitening fingerprints and validation manifest.
- W&B summary: final checkpoint
  `/workspace/ckpt/inv016_whiten_abs_recon_cov_var_ego4d_recon1/phase1_step15000.pt`, SHA-256
  `16dee2ea98aad31e231ab9bc3a88ea1393c1350a89cbe005a2fac9b3e361c66a`.
- Full interpretation: [`ANALYSIS.md`](ANALYSIS.md); architecture/measurement correction:
  [`../reconstruction_floor_architecture_audit/ANALYSIS.md`](../reconstruction_floor_architecture_audit/ANALYSIS.md).

## Conclusion

The simple hypothesis “run 058 failed because reconstruction weight 0.05 was too small” is not
supported. A 20x weight increase made reconstruction dominate 99.6% of the final scalar objective,
yet the raw floor moved only about 0.007 and exact-chunk dependence remained below 8%. The leading
open explanations are target/template structure, source-aware measurement validity, whitening and
information bandwidth, and optimization/capacity—not scalar loss weight alone.
