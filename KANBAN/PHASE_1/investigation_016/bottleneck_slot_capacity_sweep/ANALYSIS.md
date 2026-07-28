# Bottleneck slot-capacity sweep — post-run analysis

**Status:** COMPLETE — all three same-commit EGO4D arms finished and synced to W&B.

## Result in one paragraph

Increasing slot count did not clear the reconstruction floor cleanly. The late random-training
`L_recon` median improves monotonically from `0.67703` to `0.67396` to `0.66298`, but the 32-to-128
gain is only `0.01405` (2.08%), below the preregistered `0.02`/3% support threshold. More
importantly, the fixed-batch representation-health guard fails: the 32-slot arm ends marginally
healthy on the recorded batch, while 64 slots collapses severely and 128 slots recovers only to
weak specificity. The endpoint diagnostic gain is just `0.00558`. The result therefore does not
support insufficient `N_c` as the leading floor mechanism and does not justify a 256-slot arm.

## Question and decision rule

This sweep tests whether the repeated EGO4D whitened-feature reconstruction floor is primarily a
shortage of abstract slot bandwidth. It changes only `N_c` from 32 to 64 to 128 while keeping
`D_c=256`, the encoder, whitening payload, bottleneck/decoder widths, objective weights, schedule,
seed, data order, and clean source commit fixed. The abstract tensor therefore grows from 8,192 to
16,384 to 32,768 scalars per example.

The preregistered support criterion is a valid monotonic reconstruction improvement with the
128-slot arm beating the 32-slot arm by at least `0.02` absolute or 3% relative in the late primary
loss. Changes below `0.01` that are non-monotonic reject slot count as the leading floor mechanism
and explicitly stop the automatic ladder before 256 slots.

## Execution identity and launch gates

| Field | Frozen value |
|---|---|
| Git commit | `820a5b560b8668fa12452e35b1e91b170a2a91a3`, clean |
| Dataset | EGO4D, fingerprint `df36af5da6e73595473024a8e79da7f1412bd94a0b838cb0bee759b303eb6b1c` |
| Encoder | pinned V-JEPA2 ViT-L/16, feature fingerprint `372adef61c7d90a08665faf65991c1fef1f13768cdd9a6850b23393c705e1155` |
| Whitening | one shared 12,800-clip payload, fingerprint `fd00b842eebeaa2f5a83a47b82275b39182a58c91131793fd48e3502fe73d0b6` |
| Recipe | present-only, absolute whitened target, cosine reconstruction, `lambda_recon=1.0` |
| Geometry | `lambda_var=0.5`, `lambda_cov=0.01`, SIGReg/slot loss off |
| Schedule | 15,000 steps, batch 64, seed 42, reconstruction ramp 2,000 steps |
| W&B group | `inv016_bottleneck_capacity_vjepa2_whitened_recon1` |

The largest-shape resource preflight passed at physical batch 64 with a 12,298,759,168-byte CUDA
peak, 62.91 examples/s, finite loss/gradients, whitening active, present-only active, and both flow
and predicted-reconstruction losses zero. The separate largest-shape Stage 0 gate also passed with
finite loss `0.51003`, gradient norm `0.03141`, zero skipped updates, `ema_m=0.996`, and the intended
mode flags.

The normal preflight path repeatedly rescanned the already validated 169,890-file dataset identity
before touching the GPU. To avoid another redundant scan, the successful resource and Stage 0
gates reused the immediately preceding strict identity payload with the exact fingerprint above.
This optimization affected only launch validation. Every paid arm still ran its own normal strict
provenance scan and recorded the full identity in W&B.

## Run manifest

| `N_c` | W&B | State | Final checkpoint |
|---:|---|---|---|
| 32 | [`x03xlpyl`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/x03xlpyl) | finished | `phase1_step15000.pt`, SHA-256 `26c2830a08d02cc4b1131eed92872cbc77bf062a8066f2dc28d7c3199366d405` |
| 64 | [`evyokqrm`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/evyokqrm) | finished | `phase1_step15000.pt`, SHA-256 `0870e48f0c82cc574b3d9c73e3809c7aab749c1dd590481c9599405c105573b7` |
| 128 | [`7pmvxrxi`](https://wandb.ai/smahalanobis-uc-davis/hjepa-vwm/runs/7pmvxrxi) | finished | `phase1_step15000.pt`, SHA-256 `02427f8e5958ab5383cd358e77466943e6b399c2cb5bf33cd2e7622cb14abafd` |

All arms began from scratch and were executed sequentially on one A100. This changes wall-clock
layout, not the scientific recipe. The 128-slot arm passed step 500—the point beyond the synchronized
external failure that invalidated investigation 007's old capacity wave—with finite metrics, zero
skips, and stable memory. It then completed all 15,000 steps; the controller verified the final
checkpoint hash and exited with `QUEUE_DONE` and `PIPELINE_DONE`.

## Interpretation boundary for EGO4D diagnostics

The fixed EGO4D diagnostic batch consists of 16 adjacent chunks from one source UID. Consequently,
`c_cross_video_cosine`, `c_std_mean`, and `L_recon_video_gap` measure exact-chunk or within-source
specificity on this batch; they do not establish global cross-source collapse or conditioning. The
batch is identical across all three same-commit arms, so it remains a valid comparative probe of
how increasing slots changes within-source content sensitivity.

There is also a structural reason not to compare raw rank across `N_c`. `c_effective_rank` pools
`B*N_c` slot rows over the 256 feature channels, and the covariance loss uses the same kind of
batch-times-slot pooling. Learned fixed slot identities can therefore raise pooled rank or improve
pooled covariance as slots are added even when different examples' flattened codes become more
parallel. In contrast, `c_std_mean` and `c_cross_video_cosine` flatten all `N_c*D_c` coordinates per
example before comparing examples. This sweep reads those views together rather than allowing raw
rank to declare a larger bottleneck healthy.

## Reading Cycle B — 32 slots (`x03xlpyl`)

**Config:** EGO4D, present-only, 15,000 steps, absolute whitened cosine target,
`lambda_recon=1.0`, `lambda_var=0.5`, `lambda_cov=0.01`, `N_c=32`, `D_c=256`.

| Q | Result | Evidence |
|---|---|---|
| Q1 — stability and mode | Pass | All 300 training rows have `grad_skipped=0`; all diagnostics have `grad_has_nan=0`; late median grad norm is `0.1226`. Present-only and whitening remain 1, prediction/flow/`L_recon_pred` remain 0. |
| Q2 — alive and example-specific | Marginal pass on the recorded batch | Late median std `0.8059`, dead fraction `0`, and cosine `0.4735`. One late diagnostic reaches `0.5067`, so the batch-specificity margin is narrow. |
| Q3 — rich rather than cramped | Pass | Late median pooled rank `85.06`; centered within-video slot rank `30.81/31` meaningful centered directions. Attention entropy is high (`0.8426`) and remains contextual rather than a gate. |
| Q4 — reconstruction learns | Pass | Diagnostic reconstruction falls `1.0163 -> 0.6709`; late median `L_recon_present=0.67116`. Training `L_recon` late median is `0.67703`; ramp reaches 1 and decoder AGC never clips. |
| Q5 — geometry and content cooperate | Pass, with source-batch qualification | Reconstruction improves while late std/rank remain healthy and median cosine stays just below the project threshold. The within-source shuffled-code gap is small (`0.02633` late median). |
| Q6 — verdict | **Strong present representation on the recorded batch** | Stable and decodable with marginally passing fixed-batch geometry; no forecasting claim is licensed. |

## Reading Cycle B — 64 slots (`evyokqrm`)

**Config:** byte-identical scientific controls to the 32-slot arm except `N_c=64` and its derived
initialization/checkpoint identity.

| Q | Result | Evidence |
|---|---|---|
| Q1 — stability and mode | Pass | Zero skipped/NaN updates, late median grad norm `0.0911`, correct present-only/whitening flags, and no flow or predicted reconstruction. |
| Q2 — alive and example-specific | Fail on the recorded batch | Late median std falls to `0.3257`, dead fraction remains 0, and cosine rises to `0.9142`. This is severe within-source directional collapse/source-chunk invariance. |
| Q3 — rich rather than cramped | Not a rescue | Late pooled rank is still `83.13` and centered slot rank is `61.97`, showing that many slot directions coexist with an example-level specificity failure. |
| Q4 — reconstruction learns | Pass in isolation | Diagnostic reconstruction falls `1.0104 -> 0.6703`; late median `L_recon_present=0.67072`. Training `L_recon` late median is `0.67396`; decoder AGC never clips. |
| Q5 — geometry and content cooperate | Fail | The tiny reconstruction change is accompanied by much worse fixed-batch example geometry. The slightly larger gap (`0.02756` late median) does not compensate for the collapse gate. |
| Q6 — verdict | **Recorded-batch collapsed / source-chunk invariant** | Operationally valid but not a capacity win. Because the batch has one source UID, this label is not a global cross-source-collapse claim. |

The collapse is not merely an initialization artifact: at step 2,000 the 64-slot arm temporarily
reaches std `0.6859` and cosine `0.5197`, then degrades by step 5,000 to std `0.2709` and cosine
`0.9354`, finishing near the same bad equilibrium. Pooled rank and centered slot rank stay high,
which is precisely why the example-level metrics are necessary.

## Reading Cycle B — 128 slots (`7pmvxrxi`)

**Config:** byte-identical scientific controls to the other arms except `N_c=128` and its derived
initialization/checkpoint identity.

| Q | Result | Evidence |
|---|---|---|
| Q1 — stability and mode | Pass | All 300 training rows have `grad_skipped=0`; all diagnostics have `grad_has_nan=0`; late median grad norm is `0.1002`. Present-only/whitening stay 1, while prediction, flow, residual target, and `L_recon_pred` stay 0. |
| Q2 — alive and example-specific | Fail on the recorded batch | Late median std is `0.64888`, dead fraction is 0, and cosine is `0.67728`. This is healthier than 64 slots but still misses the project's `~0.8–1.2` std and `<0.5` cosine gates. |
| Q3 — rich rather than cramped | Not a rescue | Late pooled rank is `188.84` and centered slot rank is `115.89`, but both coexist with the Q2 failure. Added fixed slot directions mechanically make these quantities easier to raise. |
| Q4 — reconstruction learns | Pass in isolation | Diagnostic reconstruction falls `1.00828 -> 0.66533`; late median `L_recon_present=0.66570`. Training `L_recon` late median is `0.66298`; reconstruction scale reaches 1 and decoder AGC never clips. |
| Q5 — geometry and content cooperate | Fail | Loss improves modestly but fixed-batch specificity remains unhealthy. The final within-source gap rises to `0.03431`, about 10.0% of the learned diagnostic improvement, so most reconstruction ability still survives a wrong exact-chunk code. |
| Q6 — verdict | **Weak / source-chunk-invariant recorded-batch representation** | Operationally valid and better than 64 slots, but not a healthy capacity win and not evidence of forecasting. |

The trajectory itself demonstrates why neither early stopping nor endpoint rank is sufficient. At
step 500 the code is almost parallel across the fixed batch (`std=0.0975`, cosine `0.9872`), it
recovers to `0.5484/0.6940` at step 2,500, and it finishes around `0.6549/0.6724`. Pooled rank rises
to `189.17`, but the example-level gate never reaches the 32-slot arm's late equilibrium.

## Primary capacity comparison

| `N_c` | Scalar capacity | Late median train `L_recon` | Late median `L_recon_present` | Final diagnostic | Late std | Late cosine | Cycle-B verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 8,192 | `0.67703` | `0.67116` | `0.67091` | `0.80586` | `0.47350` | Strong on recorded batch |
| 64 | 16,384 | `0.67396` | `0.67072` | `0.67034` | `0.32574` | `0.91422` | Recorded-batch collapsed |
| 128 | 32,768 | `0.66298` | `0.66570` | `0.66533` | `0.64888` | `0.67728` | Weak/source-chunk invariant |

The 32-to-64 comparison improves the late random-training reconstruction median by only `0.00307`
and the late fixed-batch diagnostic median by only `0.00044`. The 128-slot arm adds another
`0.01098` training-median improvement, producing a total 32-to-128 gain of `0.01405` (2.08%). That
is a borderline scalar improvement in isolation, but the late fixed-batch diagnostic gain is only
`0.00546` and the final gain is `0.00558`.

Representation health prevents treating the training-median change as evidence for the capacity
hypothesis. Relative to 32 slots, 64 slots cuts late std by about 60% and raises cosine by about
0.44. The 128-slot arm recovers partway, but its std remains about 19% lower and cosine about 0.20
higher than the control. Exact-chunk conditioned share rises from 7.66% to 8.13% to 10.01%, a real
but small trend: roughly 90% of the 128-slot decoder's learned improvement still survives code
shuffling within the single source. A larger slot tensor can therefore lower loss slightly without
establishing a healthier example-specific representation.

## Final verdict and next decision

The numerical late-training delta (`0.01405`) sits in the preregistered borderline band, but the
conditional 256 rule cannot be applied without its representation-health guard. The 64- and
128-slot arms do not produce healthy fixed-batch spread/specificity, so their lower reconstruction
loss is explicitly **not a capacity win** under `PLAN.md`. Paying for 256 slots would answer whether
an unhealthy loss curve continues, not whether the bottleneck learns a healthy content code.

Do not launch 256. Reject insufficient `N_c` as the leading EGO4D reconstruction-floor mechanism
under the current whitened objective. This result does not rule out `D_c`, the 1,024-to-256 input
projection/mixer width, or whitening as bottlenecks; it specifically says that adding more learned
query slots is not the sensible next capacity lever. Proceed to the colleague-directed
no-whitening EGO4D reconstruction arm and, separately, a channel-width/`D_c` experiment with a
healthy-geometry gate. Alternate encoders remain an orthogonal substrate experiment.

---

## 2026-07-17 independent live-W&B re-audit — initialization and loss-scaling correction

This dated addendum preserves the original read above and records two control limitations found by
reconciling the downloaded W&B artifacts with an exact replay of the production constructor. The
full audit is stored at the human-requested path in investigation 015:
[`run_055.../ANALYSIS.md`](../../investigation_015/run_055_ae_latent_stack_whiten_abs_recon/ANALYSIS.md#2026-07-17-cross-investigation-addendum--live-wb-audit-of-the-ego4d-bottleneck-slot-capacity-sweep).

### What the live evidence confirms

- The API histories contain exactly 300 training rows and 30 diagnostic rows per arm.
- Provenance differs in only six leaves: `N_c`, its derived initialization/common hashes,
  checkpoint directory, and display name. Code, data/order, encoder features, whitening payload,
  schedule, and seed are genuinely controlled.
- All three downloaded whitening files are byte-identical.
- All output logs contain 300 metric records and no fatal error, warning, OOM, skipped update, or
  AGC activation.
- The 32-slot arm exactly reproduces run 060 across all 46 scientific history columns at all 300
  logged steps. This validates same-shape determinism and the 32-slot control.
- The late 32-to-128 training-median gain is real within this trajectory: `0.01404935`
  (`2.07513837%`), with the 128-slot arm lower on 259 of 260 post-warmup matched batches.

### Correction 1 — same seed did not produce identical shared weights

The description “byte-identical controls except `N_c` and its derived initialization identity” is
correct for source/config/data but too strong for parameter values. `Bottleneck.__init__` creates
the `N_c x 256` orthogonal query matrix before the latent blocks, flow, and decoder. Its
shape-dependent RNG consumption shifts later initializations.

An exact replay of commit-equivalent production code and the sweep config found:

| Module | State tensors | Identical | Shape changed | Same shape, different values |
|---|---:|---:|---:|---:|
| B | 93 | 65 | 1 | 27 |
| F_c | 70 | 28 | 2 | 40 |
| D | 59 | 29 | 0 | 30 |

`F_c` is inactive, but the differing B latent-block and D attention parameters are active. The
sentence above saying the 64-slot collapse is “not merely an initialization artifact” is therefore
superseded: its late emergence rules out only a step-zero metric illusion, not an initialization-
selected training basin. With one seed per shape, slot-count and initialization effects cannot be
separated.

### Correction 2 — fixed `lambda_cov` is not fixed effective geometry pressure

`L_cov` pools `B*N_c` slot rows over feature channels. Learned orthogonal slot identities make it
easier to satisfy as slots are added. At step zero, before reconstruction is active:

| `N_c` | Raw `L_cov` | Weighted covariance | Total objective |
|---:|---:|---:|---:|
| 32 | `6.72580338` | `0.06725803` | `0.56725800` |
| 64 | `2.93429971` | `0.02934300` | `0.52934301` |
| 128 | `0.99159271` | `0.00991593` | `0.50991595` |

Pooled rank is similarly mechanical at initialization: approximately 31, 63, and 127 while
example std is approximately zero and example cosine is one. The higher late pooled rank at 128
cannot be interpreted as content capacity without the flattened example metrics and shuffled-code
test.

### Fixed-batch baseline correction

The untrained fixed-batch losses are `1.01626515`, `1.01043403`, and `1.00827813`; the final values
are `0.67091203`, `0.67033505`, and `0.66532779`. The 128-slot arm finishes `0.00558424` below 32,
but starts `0.00798702` below it and learns `0.00240278` less improvement relative to its own
initial scaffold. This does not erase the paired training-distribution gain; it means the fixed-
batch endpoint is not an independent capacity win.

### Revised causal wording

The decision remains **do not run 256**, but the generalization is narrower:

> In this one-seed, absolute-whitened, covariance-regularized implementation, increasing `N_c`
> produces a small, coherent training-loss improvement without healthy recorded-batch specificity.
> The realized larger-slot arms fail the registered joint criterion, so more query slots are not
> the leading next lever. The experiment does not isolate activation capacity from initialization
> coupling or `N_c`-dependent geometry-loss scaling, and it does not rule out channel width,
> `D_c`, whitening, or decoder capacity.

The cheapest decisive next measurement is source-diverse checkpoint re-evaluation with verified
cross-source code derangement. After that, test the 64-slot anomaly across additional seeds or
module-local initialization streams before calling it structural. Keep the queued no-whitening
32-slot arm and a separate mixer/`D_c` experiment as the actual reconstruction-floor probes.
