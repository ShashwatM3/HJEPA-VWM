# Wave 2 - remaining mild-cov plus high-cov band

Wave 2 uses 5 GPUs. It fills the remaining mild-cov points and tests high covariance up to
`lambda_sigreg=10.0`. The omitted high-high point is `lambda_sigreg=12.5, lambda_cov=0.01`.

| GPU | W&B name / tag | `lambda_sigreg` | `lambda_cov` | W&B id | Final step | Verdict |
|---:|---|---:|---:|---|---:|---|
| 0 | `po_geom_sig7p5_cov0p003` | 7.5 | 0.003 | TBD | TBD | TBD |
| 1 | `po_geom_sig12p5_cov0p003` | 12.5 | 0.003 | TBD | TBD | TBD |
| 2 | `po_geom_sig5_cov0p01` | 5.0 | 0.01 | TBD | TBD | TBD |
| 3 | `po_geom_sig7p5_cov0p01` | 7.5 | 0.01 | TBD | TBD | TBD |
| 4 | `po_geom_sig10_cov0p01` | 10.0 | 0.01 | TBD | TBD | TBD |

Launch instructions are in [`../GUIDE.md`](../GUIDE.md), section 6.

Only launch this wave after Wave 1 reaches the early tripwires cleanly. If `lambda_cov=0.003`
already hurts reconstruction, cosine, or training stability, this wave is likely low value.
