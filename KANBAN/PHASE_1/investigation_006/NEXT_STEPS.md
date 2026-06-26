# Next steps — investigation 006

## Experiment ladder

| # | Run | `lambda_recon` | Result |
|---|---|---|---|
| 1 | [`fanciful-lake-18`](fanciful-lake-18/OBSERVATIONS.md) | 0.05 (present anchor) | **DONE.** Cliff removed; rank ceiling NOT broken (~13.3); copy gate failed (2.59). |
| 2 | [`easy-blaze-19`](easy-blaze-19/OBSERVATIONS.md) | 0.05 present + **0.05 predicted (option 3, through-`F_c`)** | **DONE.** Stable full 15k, but copy gate unchanged / worse; `L_recon_pred` flat; recon floor identified. |

**Decision after run 2:** do **not** repeat option 3 at the same capacity. The joint
objective was implemented faithfully and ran cleanly, but the reconstruction channel is
saturated near relative MSE ~0.60 and cannot transmit prediction-quality information to
`F_c`. Full reasoning in [`easy-blaze-19/OBSERVATIONS.md`](easy-blaze-19/OBSERVATIONS.md).

The remaining fork is no longer present-vs-predicted reconstruction. It is whether to
first make reconstruction unsaturated enough to carry future-prediction error, or to
change the prediction target / horizon because copy-forward may be too strong at
`horizon_k=12`.

---

## Candidate follow-up investigations

### A. Break the reconstruction floor

Open a new investigation if the team still wants reconstruction to train prediction.
The run question should be explicit: can any decoder / bottleneck setting move the
readouts below the ~0.60 floor enough that `L_recon_chat` and `L_recon_cplus` separate?

Possible levers:
- larger `c` capacity,
- larger / deeper decoder `D`,
- higher `lambda_recon` / `lambda_recon_pred`,
- a controlled capacity ablation to locate whether the floor is in `c`, `D`, or both.

Only after the floor moves should option 3 be retried.

### B. Revisit the copy gate target

Open a separate investigation if the team believes horizon-12 copy is intrinsically too
strong. `coarse_copy_loss` keeps falling late, so the prediction failure may be a target /
horizon property rather than a missing reconstruction branch. This would point toward
changing horizon design or the prediction target, not another decoder-side auxiliary
loss.

---

## Do not do next

- Do not rerun `lambda_recon=0.05` + `lambda_recon_pred=0.05` as-is.
- Do not spend a run only on a "clean" option-3 variant before addressing the
  reconstruction floor. The easy-blaze result says the current channel is blind even
  when trained through `F_c`.
- Do not record option 3 as a stability failure. Stability held; the failure was lack
  of useful signal.
