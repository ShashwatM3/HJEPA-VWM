# Observations — DINOv3 latent-shape sweep

The exact center has completed as W&B `fiactcw6`; no non-center shape has launched.

Compared with no-geometry Run 69, center `fiactcw6` raises effective rank `14.60 -> 69.00`,
centered slot rank `10.51 -> 29.03`, and std `0.3329 -> 0.6222`, while reducing within-source
cosine `0.8699 -> 0.6849`. Covariance plus variance is a real geometry lever but does not reach
the recorded-batch spread/alignment thresholds. The center verdict is **COLLAPSED REP ON THE
RECORDED WITHIN-SOURCE BATCH; GLOBAL COLLAPSE INDETERMINATE**.

Full center evidence:
[`../../investigation_016/run_071_unwhitened_dinov3_m512_cov_var/`](../../investigation_016/run_071_unwhitened_dinov3_m512_cov_var/).
SigLIP 2 has the same native tensor shape but a different target space, so its winner is not
assumed to transfer to DINOv3.

W&B IDs, states, late-window measurements, and the lane verdict for the eight contrasts will be
appended after launch.
