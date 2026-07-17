# Observations — bottleneck slot-capacity sweep

## 2026-07-17 — complete same-commit EGO4D result

All three arms finished cleanly with zero skipped/NaN updates, correct present-only flags, final
checkpoints, and W&B sync. Their only scientific configuration delta is `N_c=32/64/128`; all share
commit `820a5b5`, seed/data order, V-JEPA feature fingerprint, whitening payload, and fixed
diagnostic batch.

| `N_c` | Late median train `L_recon` | Late median fixed `L_recon_present` | Late std | Late cosine | Verdict |
|---:|---:|---:|---:|---:|---|
| 32 | `0.67703` | `0.67116` | `0.80586` | `0.47350` | Strong on recorded batch, marginal geometry margin |
| 64 | `0.67396` | `0.67072` | `0.32574` | `0.91422` | Recorded-batch collapsed/source-chunk invariant |
| 128 | `0.66298` | `0.66570` | `0.64888` | `0.67728` | Partial recovery, still fails recorded-batch geometry |

The primary training median improves monotonically, but 128 beats 32 by only `0.01405` (2.08%),
short of the `0.02`/3% capacity-support threshold. The fixed diagnostic gain is only `0.00546`.
Exact-chunk conditioned share rises from 7.66% to 8.13% to 10.01%, while most decoder improvement
still survives a wrong within-source code.

Raw pooled rank is misleading across this sweep: late rank is about `85/83/189`, and centered slot
rank rises mechanically with slot count, even while example-level std/cosine fail at 64 and 128.
This matches the code semantics: rank/covariance pool batch-times-slot rows, whereas std and cosine
compare flattened examples. More fixed slot directions are not proof of more input-dependent
content.

The conditional 256 arm is not warranted. Although the training-loss delta alone is numerically
borderline, the preregistered health guard says lower loss with unhealthy spread is not a capacity
win. The next paid axes are no whitening and the 1,024-to-256 channel/`D_c` bottleneck, not more
query slots. See [`ANALYSIS.md`](ANALYSIS.md) for the complete Reading Cycle B record.
