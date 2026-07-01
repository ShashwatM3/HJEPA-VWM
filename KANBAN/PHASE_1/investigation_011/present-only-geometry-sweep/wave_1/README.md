# Wave 1 - SIGReg ladder plus mild-cov anchors

Wave 1 uses 5 GPUs. It finishes the SIGReg-only ladder beyond the existing anchor and samples mild
covariance at two informative points.

Existing anchor, not relaunched:

| W&B name | W&B id | `lambda_sigreg` | `lambda_cov` | Final rank | Verdict |
|---|---|---:|---:|---:|---|
| `inv011_fixed_position_present_recon` | `hcr2qx19` | 5.0 | 0.0 | 49.60 | Existing anchor/control |

| GPU | W&B name / tag | `lambda_sigreg` | `lambda_cov` | W&B id | Final step | Verdict |
|---:|---|---:|---:|---|---:|---|
| 0 | `po_geom_sig7p5_cov0` | 7.5 | 0.0 | TBD | TBD | TBD |
| 1 | `po_geom_sig10_cov0` | 10.0 | 0.0 | TBD | TBD | TBD |
| 2 | `po_geom_sig12p5_cov0` | 12.5 | 0.0 | TBD | TBD | TBD |
| 3 | `po_geom_sig5_cov0p003` | 5.0 | 0.003 | TBD | TBD | TBD |
| 4 | `po_geom_sig10_cov0p003` | 10.0 | 0.003 | TBD | TBD | TBD |

Launch instructions are in [`../GUIDE.md`](../GUIDE.md), section 4.

Wave 1 decides whether:

- stronger SIGReg alone clears `c_effective_rank >= 60`;
- mild covariance helps at the current anchor (`5.0`) or likely useful band (`10.0`);
- any covariance pressure causes early instability before trying the `0.01` band.
