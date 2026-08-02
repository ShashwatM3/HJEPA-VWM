# Observations — `ihiuptdp`

The run reached step 7,100 with every logged `grad_skipped`, `grad_has_nan`, and
`instability_warn` value equal to zero. Late six-diagnostic medians are:

| Metric | Value |
|---|---:|
| fixed correct-code reconstruction | 0.38514 |
| fixed rolled-code reconstruction | 0.42271 |
| exact-chunk gap | 0.03759 |
| `c_std_mean` | 0.71049 |
| within-source pair cosine | 0.55934 |
| `c_dead_dim_frac` | 0 |
| `c_effective_rank` | 64.81 |
| centered slot rank | 14.64 / 15 |

The partial arm was stable and code-dependent, with rank above 60 and near-maximal centered slot
rank for 16 slots. Spread and pair separation had not reached the project bands when the run
ended. Verdict: **INVALID AS A COMPLETE SWEEP ARM; STABLE, DECODABLE PARTIAL TRAJECTORY**.

The single-source EGO4D batch means the pair/gap evidence remains within-source. These values must
not be substituted for a final `16×512` result.
