# Two-step rollout change and expectation audit

**Audit date:** 2026-08-20
**Audited branch:** `exp/two-step-rollout-loss`
**Baseline:** `df5d959b76038c451ab74218bfa1050c996af6e5`
**Implementation:** `db69f3a` (`feat: add two-step rollout endpoint loss`)
**Current HEAD:** `ddf310f2d0b07405f393c521adb771c029e6a240`
(`refactor: focus rollout W&B metrics`)

## Executive verdict

The current code implements a genuine differentiable, two-evaluation Euler rollout objective for
one deliberately narrow regime: fixed-bottleneck, present-source, `fc_only` training. Starting from
the detached present latent, it evaluates the existing `CoarseFlow` at times 0 and 0.5, takes two
steps of size 0.5, passes the first generated state into the second call, and applies endpoint MSE
only against the detached future latent. Both flow calls and their connecting midpoint remain in
one autograd graph; configuration validation ensures only `F_c` is trainable.

The intervention does not change the encoder, bottleneck, flow architecture, data, original
teacher-forced objective, optimizer, clipping, inference solver, or checkpoint tensor schema. It
adds two `F_c` calls per treatment update and adds their ramped endpoint loss to the original
flow-matching loss. A zero weight skips those calls and preserves the training math, RNG draws,
gradients, and optimizer update of the baseline path. It does **not** preserve all baseline program
behavior byte-for-byte: the config/checkpoint/provenance dictionaries gain two fields, the train-step
metric dictionary gains zero-valued rollout keys, and `ddf310f` changes W&B from logging the whole
metric dictionary to a fixed allowlist.

Scientifically, this is a direct test of the teacher-forced-state versus generated-rollout-state
gap observed before the prior run's late collapse. It is not a causal test of the distinct
step-5350--5450 gradient/AGC event, and success on this objective cannot establish human-like future
understanding.

## Evidence labels

- **Code fact**: verified in the executable source at current HEAD or its Git diff from baseline.
- **Prior-run evidence**: measured values already recorded in repository forensic reports/W&B
  readouts; not re-measured in this audit.
- **Expectation**: a prediction for the proposed matched run, not an observed result.
- **Unknown**: requires actual control/treatment training or resource profiling.

## 1. Repository and change-set forensics

### Git state

At audit time the branch was `exp/two-step-rollout-loss`, HEAD was `ddf310f`, and `git status
--short --branch` showed no working-tree changes. Exactly two commits followed the requested
baseline:

1. `db69f3a` — implementation, tests, config/docs, and several pre-existing investigation reports
   committed together.
2. `ddf310f` — reduced/renamed rollout metrics, introduced a W&B allowlist, updated the history
   exporter and tests/docs.

### Every file changed after the baseline

| File | Commit(s) | Meaningful change |
|---|---|---|
| `config.py` | `db69f3a` | Added non-negative `TrainConfig.lambda_rollout` (default 0) and `rollout_ramp_steps` (default 1500). |
| `configs/train.yaml` | `db69f3a` | Added the same two fields with zero/default values. No other recipe value changed. |
| `train.py` | both | Added the K=2 Euler helper, gated rollout loss/metrics, validation, CLI overrides, and tests-facing return values. Cleanup renamed/reduced rollout metrics and added the W&B allowlist. |
| `tests/test_rollout_loss.py` | both | Added contracts for midpoint gradient/time grid, loss weighting and parameter scope, zero-weight call avoidance, config/CLI validation, and final W&B schema. Cleanup updated metric contracts and added allowlist coverage. |
| `run_history.py` | `ddf310f` | Updated export lists for active losses, stability, fixed rollout metrics, conditioning diagnostics, and recon readouts; removed several redundant/raw diagnostic fields from default export. This does not affect training. |
| `AGENT_FILES/AGENTS.md` | both | Documented the new endpoint objective, gradient scope, total equation/defaults, zero behavior, and focused W&B contract. Documentation is corroborating context, not the source of truth. |
| `GUIDES/CODEBASE_STRUCTURE.md` | `db69f3a` | Added the new rollout-loss test file to the repository map. |
| `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md` | both | Added rollout-objective semantics, then replaced verbose/dynamic W&B guidance with the final fixed decision metrics. |
| `GUIDES/READING_EXPERIMENTS.md` | both | Added rollout config to the run-reading precheck and documented final logging cadence/schema. |
| `FC_ARCHITECTURE_AND_CONDITIONING_INVESTIGATION.md` | `db69f3a` | Added a read-only historical investigation of `F_c` architecture/conditioning. No executable behavior. |
| `FLOW_MATCHING_MECHANISM_FORENSIC_INVESTIGATION.md` | `db69f3a` | Added the baseline mechanism and prior-run failure audit that contains the step-5000 and step-5350--5500 evidence. No executable behavior. |
| `NOISELESS_FLOW_PRECONDITION_VALIDATION.md` | `db69f3a` | Added a prior precondition/slot-alignment investigation. No executable behavior. |
| `NOISELESS_PRESENT_TO_FUTURE_FLOW_INVESTIGATION.md` | `db69f3a` | Added the earlier present-source design investigation. No executable behavior. |
| `PRECOLLAPSE_ROLLOUT_GAP_INTERVENTION_PLAN.md` | `db69f3a` | Added the proposal and historical thresholds underlying this implementation. Several proposed names differ from final code; it must not be treated as executable truth. |
| `RUN040_HISTORICAL_ENVIRONMENT_RECOVERY.md` | `db69f3a` | Added historical checkpoint/environment recovery notes. No executable behavior. |
| `RUN60_DINOV3_NOISELESS_FLOW_VALIDATION.md` | `db69f3a` | Added Run-60 fixed-bottleneck validation notes. No executable behavior. |
| `WANDB_INV019_BOTTLENECK_SWEEP_ANALYSIS.md` | `db69f3a` | Added prior W&B sweep analysis. No executable behavior. |
| `WANDB_INV019_BOTTLENECK_SWEEP_ANALYSIS_ERRATUM.md` | `db69f3a` | Added corrections to that analysis. No executable behavior. |

No changes occurred in `models.py`, `losses.py`, `data.py`, `diagnostics.py`, `encoders.py`, or the
checkpoint implementation itself.

## 2. Exact loss and tensor path

Let batch size be `B`, coarse slots `N_c`, and coarse width `D_c`. The intended experiment uses
`N_c=64`, `D_c=512`; all equations below hold for any valid configured shape.

### Existing present-source teacher-forced loss

The fixed saved online bottleneck is evaluated under `torch.no_grad()` on both clips:

```text
context_clip x_t       : (B, 8, 3, 256, 256)
future_clip  x_{t+k}   : (B, 8, 3, 256, 256)
e_t     = E(x_t)       : (B, N_e, D_e), no grad
e_plus  = E(x_{t+k})   : (B, N_e, D_e), no grad
c_0     = B_fixed(e_t) : (B, N_c, D_c), detached
c_1     = B_fixed(e+)  : (B, N_c, D_c), detached
```

For each example, code samples `tau ~ Uniform[0,1)` from its purpose-specific deterministic stream
and a condition-drop mask. In the matched experiment dropout is configured as zero:

```text
z_tau = (1 - tau)c_0 + tau c_1                         (B, N_c, D_c)
u     = c_1 - c_0                                      (B, N_c, D_c)
u_hat = F_c(z_tau, tau, condition=c_0, condition_drop) (B, N_c, D_c)

L_flow = mean_{b,n,d} (u_hat - u)^2
```

This is the old optimized equation in `fc_only`:

```text
L_old = L_flow.
```

The state `z_tau` contains the realized future with coefficient `tau` whenever `tau>0`; this is
the teacher-forced path.

### New differentiable two-step endpoint path

The same detached `c_0` is both initial state and explicit condition. The helper constructs no
future argument:

```text
tau_0 = zeros(B)                         # exact time 0
v_0   = F_c(c_0, tau_0, c_0, no_drop)   # (B, N_c, D_c)
d_0   = 0.5 v_0                          # first Euler displacement
c_1/2_hat = c_0 + d_0                    # generated midpoint, (B, N_c, D_c)

tau_1 = full(B, 0.5)                     # exact time 0.5
v_1   = F_c(c_1/2_hat, tau_1, c_0, no_drop)
d_1   = 0.5 v_1
c_hat_1 = c_1/2_hat + d_1                # generated endpoint

L_rollout = mean_{b,n,d} (c_hat_1 - c_1)^2.
```

Thus “two-step rollout” means **two forward-Euler velocity evaluations over `[0,1]`, each with
step size `h=0.5`**, not a two-frame horizon, two target clips, two separately supervised steps, or
two calls on teacher-forced states. The second call consumes the first predicted state. Both calls
reuse the present condition `c_0`. They forcibly receive an all-false drop mask, so the rollout
term never uses the learned null condition even if the ordinary flow call's configured condition
dropout were nonzero.

The new optimized equation is:

```text
r(s) = 1                                  if rollout_ramp_steps <= 0
     = clamp(s / rollout_ramp_steps, 0, 1) otherwise

lambda_eff(s) = lambda_rollout * r(s)
L_new(s) = L_flow + lambda_eff(s) L_rollout.
```

For the requested treatment (`lambda_rollout=0.10`, `rollout_ramp_steps=1500`):

| Zero-indexed global step | Ramp | Effective weight |
|---:|---:|---:|
| 0 | 0 | 0 |
| 500 | 1/3 | 0.0333333 |
| 1500 | 1 | 0.10 |
| 2500 | 1 | 0.10 |
| 5000 | 1 | 0.10 |

The call gate is `lambda_rollout > 0`, not `lambda_eff > 0`. Therefore the treatment executes both
extra calls even at step 0 although their weighted contribution is exactly zero. The control with
configured `lambda_rollout=0` executes neither.

### Gradient routing

`c_0` and `c_1` come from `_fixed_flow_forward`, whose encoder and fixed bottleneck work is entirely
inside `torch.no_grad()`. Configuration validation additionally requires `optimization_scope=fc_only`;
the optimizer contains only `F_c` parameters and B/B_EMA/D are frozen.

For the rollout term:

```text
dL_rollout/dtheta_Fc
  = direct contribution through F_c(c_1/2_hat, 0.5, c_0)
  + contribution through c_1/2_hat and F_c(c_0, 0, c_0).
```

There is no detach between the calls. Consequently all `CoarseFlow` parameters used in either call
(slot/type/null parameters as applicable, time MLP, all AdaLN blocks, and final LayerNorm) can
receive gradient. The explicit all-false mask means the learned `null_condition` is selected by no
examples and should receive a zero/absent useful gradient from this rollout branch; it may still be
trained by the ordinary teacher-forced call if dropout is nonzero. Encoder, fixed B, B_EMA, and D
receive no gradients and are absent from the optimizer. In the intended zero-drop matched arms,
only the non-null `F_c` path is trained by either loss.

### Future-target leakage audit

The future latent `c_1` enters:

1. the original teacher-forced `z_tau` and target velocity `u`; and
2. the final rollout endpoint MSE and its detached logging calculations.

It does **not** enter the rollout initial state, midpoint, explicit condition, times, dropout mask,
or either `F_c` input through any other route. The helper's signature has no future tensor. This
confirms no accidental rollout-path leakage while preserving the intentional future exposure of the
original teacher-forced term.

### Zero-weight compatibility

With `lambda_rollout=0`, the conditional block is skipped. There are no additional `F_c` calls,
no additional random draws, no rollout activation graph, and no additional loss term. The old
`L_flow` computation, backward graph, AGC/global clipping, and optimizer update remain numerically
the baseline path. The added test monkeypatches the helper to fail if invoked and verifies it is not
called.

Important qualifications:

- The test checks call avoidance and zero metrics, but it does not perform the plan's stronger
  byte-for-byte comparison of RNG state, parameter gradients, and optimizer state against the
  baseline commit.
- Current HEAD always returns six zero rollout metrics from `train_step` when disabled.
- Current HEAD always filters W&B through `_select_wandb_metrics`; this changes logging behavior
  from the baseline even in a zero-weight run.
- Saved config/provenance now contains the two new config fields. Model and optimizer state formats
  are otherwise unchanged.

Therefore “baseline numerical training behavior is preserved” is supported by code and targeted
contract tests; “complete execution/output behavior is identical” is false.

## 3. What remained unchanged

| Area | Verdict | Verification |
|---|---|---|
| Encoder and bottleneck | Unchanged implementation | No diff in `encoders.py` or `models.py`. Active rollout is restricted to one frozen saved online bottleneck for both endpoints. |
| Flow-model architecture | Unchanged | `CoarseFlow` is still the same concatenated state/condition transformer with time MLP, AdaLN blocks, and final LayerNorm. Only call sites/loss changed. |
| Present/future latent construction | Unchanged | `_fixed_flow_forward` still runs the same frozen encoder and same fixed bottleneck under `no_grad()` on context and target. |
| Original flow-matching target | Unchanged | `u=c_1-c_0`; `z_tau=(1-tau)c_0+tau c_1`; velocity MSE is retained in full. |
| Optimizer | Unchanged | Same AdamW creation, beta values, decay partition, and `fc_only` parameter selection. New config fields do not enter optimizer construction. |
| Learning-rate schedule | Unchanged | Same linear warmup/cosine decay. The rollout ramp changes loss weight only, not LR. |
| AGC and global clipping | Unchanged | Same per-module AGC, `grad_clip=0.5`, skip threshold, and ordering. Treatment gradients feed the same machinery, so observed activity may change. |
| Dataset and sampling | Unchanged | No `data.py` diff. Same clip discovery, deterministic per-sample train window selection, shared augmentations, and sampler/resume logic. |
| Horizon/frame spacing | Unchanged by commits | YAML remains `horizon_k=12`, stride 2. The planned matched experiment must explicitly override horizon to 16; that is a run config, not this code change. |
| Inference/evaluation solver | Unchanged | Existing diagnostic `flow_euler_rollouts` was not modified. The new helper is a training-loss integrator, not a replacement inference solver. |
| Checkpoint behavior | Tensor/state behavior unchanged | Save/load code is unchanged; checkpoints still save all modules, optimizer, RNG/sampler, config and provenance. Config payloads now naturally include the two new fields. |
| Condition dropout | Base mechanism/config unchanged | YAML still defaults to 0.10. Ordinary flow uses the configured sampled mask; the new rollout calls explicitly force no drop. The planned arms must override to 0. |
| Precision | Unchanged | YAML remains bf16; rollout runs inside the same training autocast block. No separate fp32 integration was added. |
| Batch configuration | Unchanged by commits | YAML remains global batch 64. Treatment may require a common reduction for both arms if resource preflight fails; no such reduction is implemented automatically. |
| Reconstruction/geometry losses | Code unchanged | Active rollout validation requires `fc_only`, so these are not optimized. Run commands should set their lambdas to zero to avoid irrelevant compute/logging ambiguity. |

The current checked-in YAML is not by itself the treatment recipe: it is `joint`, horizon 12,
condition dropout 0.10, `lambda_recon=1`, and `lambda_rollout=0`. The intended experiment requires
explicit overrides to fixed-present `fc_only`, horizon 16, dropout 0, inactive B/D losses, and the
selected rollout weight.

## 4. Final W&B logging behavior

Current HEAD combines train-step and (every `diag_every`) diagnostic dictionaries, then applies a
fixed allowlist. Keys are emitted only if present in that iteration and, for several objectives,
only if enabled by config.

### Existing baseline/core metrics retained

Always eligible:

- Optimization: `loss`, `L_flow`, `lr_mult`, `grad_norm`, `grad_skipped`, `grad_has_nan`,
  `instability_warn`, `agc_Fc_clipped`, `agc_Fc_max_ratio`.
- Collapse/representation: `c_std_mean`, `c_cross_video_cosine`, `c_effective_rank`.
- Prediction/copy/conditioning: `coarse_copy_loss`, `coarse_vs_copy_ratio`,
  `coarse_vs_batch_mean_ratio`, `coarse_condition_shuffle_degradation`, plus the four
  `teacher_forced_random_tau_*` copy/batch-mean/shuffle keys.
- `c_plus_effective_rank` only when no fixed bottleneck checkpoint is configured.
- `agc_B_clipped` and `agc_B_max_ratio` only in joint scope.
- Active-objective metrics only when their weights are positive: `L_var`, `L_cov`, `L_slot`,
  `L_sigreg`/`sigreg_scale`, and the relevant reconstruction losses/scales/readouts/AGC metrics.

For the intended fixed-present `fc_only` arms with representation/reconstruction lambdas explicitly
zero, `c_plus_effective_rank`, B/D AGC, and inactive auxiliary-loss metrics are not logged. The
remaining fixed representation metrics are useful as frozen-state/provenance/collapse guards even
though they should not move.

### Newly introduced rollout metrics

| Metric | Decision supported |
|---|---|
| `loss/rollout` | Is the generated K=2 endpoint objective being fitted? Raw MSE, unweighted. |
| `rollout/lambda_effective` | Is the intended 0→0.10 ramp active at the expected step? |
| `rollout/copy_ratio` | Does the training-batch K=2 endpoint beat the do-nothing present endpoint? Primary intervention-quality ratio. |
| `rollout/displacement_cosine` | Is predicted temporal change pointed toward the true change? Separates directional failure from scalar MSE alone. |
| `rollout/displacement_norm_ratio` | Is predicted motion too small/large relative to true motion? Separates magnitude failure from direction. |
| `rollout/target_displacement_valid` | Is the copy denominator/target direction large enough (`copy_mse>1e-8`) for the previous three geometric ratios to be meaningful? |

These six keys are present as zeros when the rollout is disabled. When active, invalid target
displacement makes copy ratio, cosine, and norm ratio `NaN` and sets the validity flag to zero.

### Remaining duplication or avoidable logging

- `loss` is algebraically derivable from `L_flow`, `loss/rollout`, and
  `rollout/lambda_effective` in the intended `fc_only` experiment. It remains worthwhile as the
  exact scalar sent to backward and for detecting unanticipated active terms.
- `rollout/lambda_effective` is derivable from config and step, but is a cheap run-integrity check.
- `rollout/target_displacement_valid` is logically derivable from an unlogged copy MSE, so it is
  necessary in the focused schema.
- `rollout/copy_ratio` and `coarse_vs_copy_ratio` are not duplicates: the former is the K=2
  generated endpoint on the live training batch; the latter is the existing diagnostic estimator
  and may use different state/evaluation semantics/cadence.
- The `teacher_forced_random_tau_*` metrics and ordinary `coarse_*` variants overlap conceptually.
  They are defensible only because the experiment explicitly needs teacher-forced versus genuine
  rollout and condition-use comparisons. Panel them deliberately rather than treating all as
  independent outcomes.
- `agc_Fc_clipped` (count) and `agc_Fc_max_ratio` answer different questions: breadth of clipping
  and worst relative spike. Neither is redundant with `grad_norm`, which is measured after AGC and
  before global clipping rescales gradients.
- The selector can still log inactive-in-practice B/D objective metrics if their lambdas are left
  positive while using `fc_only`. The matched commands should set those lambdas to zero. This is a
  configuration hygiene issue, not a dynamic-key explosion.
- `run_history.py` retains inactive auxiliary loss names in its requested `CORE_METRICS`; absent
  W&B columns cost clarity in exported schemas but do not create training series.

The cleanup successfully removes dynamic K/condition series, raw first/second step norms,
duplicated weighted loss, post-clip norm, fixed parameter count, and boolean/count variants that did
not change the experiment decision.

## 5. What the experiment tests

### Hypothesis

The retained teacher-forced objective samples only paired straight-line states

```text
x_tau = (1-tau)c_0 + tau c_1,
```

and fits the constant target vector `c_1-c_0`. At every nonzero `tau`, `x_tau` contains some exact
realized-future information. Inference instead starts at `c_0` and constructs later states from
earlier predicted velocities. A field that is excellent on paired interpolation states can
therefore fail on its own generated states.

**Prior-run evidence:** at step 5000 the teacher-forced endpoint/copy ratio was `0.08842`, while
genuine Euler rollout ratios were `1.04453` (one evaluation) through `1.07099` (eight). The K=2
value was `1.04914`; direction cosine was about `0.203`, and predicted displacement norm was about
`0.57x` the true norm. This is strong evidence of a teacher-forced/generated-state generalization
gap, though it does not quantify a literal manifold distance.

The new loss exposes the model to exactly one self-generated intermediate state
`c_1/2_hat` during training and penalizes its final generated endpoint against the true future. It
does not provide an artificial target velocity at that intermediate state; endpoint supervision
lets gradients choose corrections through both calls.

### What it does not test directly

The prior run's first optimization warning was at step 5350 (`grad_norm=15.3487`, six `F_c` tensors
AGC-clipped), the largest excursion at 5450 (`grad_norm=38.9573`, eight clipped), and the first
diagnostic conditioning/predictor collapse at 5500. Temporal adjacency does not prove causation.
This experiment ends/evaluates at 5000 and primarily addresses the already-existing pre-collapse
exposure gap. Rollout gradients could improve or worsen later stability, but the design does not
isolate the cause of the 5350--5450 event.

## 6. Expected outcomes at checkpoints 2500 and 5000

### Matched-control expectations

**Control (`lambda_rollout=0`)** should reproduce the prior fixed-present training trajectory within
normal hardware/data-order numerical tolerance if commit, checkpoint/hash, F_c initialization,
dataset identity/order, seed, batch, horizon, precision and all remaining flags are identical. It
should show no K=2 training calls, zero rollout training metrics, and the historical pattern:
teacher-forced performance improves strongly while genuine rollout remains around or above copy.

The strongest available checkpoint anchors are the prior genuine rollouts:

| Prior checkpoint | K=2 endpoint/copy | K=2 displacement cosine | K=2 predicted norm / true norm |
|---:|---:|---:|---:|
| 2500 | 1.1110 | 0.2675 | 137.021 / 181.838 = 0.7535 |
| 5000 | 1.0491 | 0.2031 | 103.849 / 181.838 = 0.5711 |

These are evidence-based reference values, not deterministic equality thresholds for a new run.

### Treatment expectations

**Treatment (`lambda_rollout=0.10`, ramp 1500)** has full rollout weight by step 2500. If exposure
is the actionable cause, its training K=2 endpoint loss and ratio should be below control at 2500
and remain improved at 5000; direction should rise and magnitude should not collapse toward zero.
The original flow loss may worsen modestly because the same field is now optimized on two state
distributions. Total loss is expected to exceed `L_flow` by `0.1*L_rollout` after the ramp, so raw
total losses must not be compared between control and treatment as if they had the same objective.

### Decision table

| Metric | Desired direction | Improvement means | No change means | Deterioration means |
|---|---|---|---|---|
| Teacher-forced `L_flow` / teacher-forced endpoint | Flat to modestly higher is tolerable; lower is welcome | On-path fit was preserved or improved while adding generated-state support | Added loss is orthogonal to on-path fit | Large worsening means objective conflict/domination; prior plan allows at most 20% teacher-forced error worsening for success |
| `loss/rollout` | Down over treatment training | The exact K=2 endpoint objective is being optimized | Weight/gradient may be ineffective, or task capacity is limiting | Optimization instability or competing losses dominate |
| `rollout/copy_ratio` | Down; below 1 | Generated endpoint beats copying present | Exposure loss has not changed practical endpoint quality | Above control or rising means rollout training is harmful/unstable |
| 4/8-step genuine rollout/copy | Down; hard target below 1 at both checkpoints | Training on one generated midpoint generalizes to deployed longer rollouts | K=2 overfits its own solver grid or exposure is not dominant | Error amplification/off-grid generalization worsened |
| Displacement cosine | Up | Predicted change points more toward true future; prior step-5000 was ~0.20 | Direction failure persists | Lower/negative means increasingly wrong direction |
| Displacement norm ratio | Toward 1 without overshoot | Motion magnitude better matches target | Existing underprediction persists | Near 0 is shortcut copying; >1 with poor cosine is energetic wrong-way motion |
| `grad_norm` | Finite, comparable to control, no late cliff | Added graph remains optimizable | Intervention did not materially affect optimization health | Sustained/spiking increase, especially with AGC, warns of rollout-loss-driven instability |
| `agc_Fc_clipped`, `agc_Fc_max_ratio` | Sparse/comparable to control | No concentrated parameter-level gradient stress | Same stability regime | Sustained broader/higher clipping means rollout gradients are being truncated or destabilizing F_c |
| Total `loss` | Down within each arm only | Each arm fits its own objective | Plateau | Rise may indicate instability, but cross-arm absolute comparison is invalid because objectives differ |
| Training stability | No NaN, skip, warning, or frozen-state change | Valid experiment | If metrics simply plateau, experiment is valid but ineffective | Any skip/NaN/frozen hash change invalidates conclusions after that point |
| Condition-shuffle degradation | Positive; prior plan asks correct MSE ≥10% better than shuffled | `F_c` uses the actual video's condition | Endpoint gain may be generic/unconditioned | Loss of gap suggests shortcut/generic prediction or conditioning collapse |
| Representation metrics | Numerically invariant between arms | Fixed-bottleneck contract held | Tiny numeric noise only | Material movement is an implementation/provenance defect, not learned representation change |

### Success, partial success, failure, and stop rules

**Success (pre-registered, evidence-derived):** at both 2500 and 5000, treatment 4- and 8-step
endpoint/copy and endpoint/batch-mean ratios are below 1 and better than control; displacement
cosine is at least 0.30 and at least +0.05 above control; norm ratio is within the provisional
historical band `[0.40, 1.25]` and not near zero; correct-condition endpoint MSE is at least 10%
below shuffled; teacher-forced-to-rollout degradation improves at least 30% while teacher-forced
error worsens no more than 20%; and there are no skips, NaNs, warnings, or frozen-state changes.

The hard `<1` and stability gates and relative control comparison are well motivated by the copy and
batch-mean baselines. The `0.30`, `+0.05`, `[0.40,1.25]`, 10%, 30%, and 20% thresholds come from the
existing intervention plan and historical ~0.20 direction/~0.57 norm observations, but that plan
explicitly labels several as provisional. The repository does not contain a broad enough
distribution to claim they are universal acceptance thresholds.

**Partial success:** treatment clearly improves K=2 and/or 4/8-step rollout versus control at both
checkpoints but misses one absolute gate, or improves at 2500 and retains some benefit at 5000
without a stability failure. This supports exposure as one contributor, not a complete solution.

**Failure:** rollout training loss falls but genuine fixed-batch 4/8-step ratios do not improve over
control at either checkpoint; treatment and control move equally; treatment worsens conditioning,
direction, or stability; or apparent gain occurs at only one lucky checkpoint. That weakens/falsifies
rollout exposure as the dominant actionable cause for this setup.

**Stop/invalid criterion:** stop on any NaN, skipped optimizer update, instability warning, material
frozen-state hash/representation change, treatment resource failure, or sustained gradient/AGC
behavior grossly outside the control envelope. The repository supplies no evidence-based exact
numeric “grossly outside” threshold beyond existing `instability_warn_grad_norm=30` (jointly with
`L_flow>1`) and `grad_skip_threshold=150`; use matched-control distributions rather than inventing
another cap. A single AGC clip is not itself a stop: the prior first event at 5350 preceded collapse
but was not proven causal.

## 7. Important interpretations

**If rollout loss decreases but copy ratio remains near or above 1, did it work?**
It optimized its auxiliary scalar but failed the practical prediction gate. If it is lower than
control yet still ≥1, call it partial optimization progress, not successful forecasting. If 4/8-step
ratios also do not improve, the exposure intervention failed as the dominant remedy.

**If copy ratio improves but teacher-forced flow loss worsens slightly, is that acceptable?**
Yes, provided the rollout gain persists at both checkpoints, conditioning/stability remain healthy,
and teacher-forced degradation stays within the pre-registered tolerance (the plan uses 20%). The
experiment is meant to trade some on-path specialization for generated-state competence.

**If displacement cosine improves but norm ratio remains poor, what does that mean?**
Direction learned before magnitude. A low ratio means the model points toward the future but moves
too little (copy-like under-motion); a high ratio means it overshoots. Endpoint MSE may remain poor
despite better semantics of direction.

**If both control and treatment improve equally, what can we conclude?**
The improvement cannot be attributed to rollout loss. It is due to shared training trajectory,
evaluation variance, or another common factor; the matched comparison is null.

**If treatment helps at 2500 but collapses by 5000, what is the likely interpretation?**
Rollout exposure produces an early useful gradient but does not yield a stable optimum at this
weight/schedule, or it accelerates a pre-existing optimization instability. Compare gradient/AGC
timing and condition-shuffle degradation with control. It does not by itself prove the old 5350
event's mechanism, especially because this collapse occurs in a modified objective.

**Can this prove the latent represents human-like future understanding?**
No. It can show improved deterministic prediction in one frozen latent space, relative to copy and
batch-mean baselines, on one dataset/horizon. It does not establish causal, compositional,
counterfactual, semantic, or human-like understanding.

**What result justifies frame skipping or a longer horizon next?**
A stable treatment that beats copy/batch mean, uses correct conditioning, and improves over control
at both checkpoints shows the model can exploit rollout exposure at the current horizon. If the
remaining limitation is small true displacement/static overlap rather than optimizer failure, then
larger frame spacing or horizon is a clean next test. Horizon 16 already removes overlap for eight
frames at stride 2 (`(8-1)*2=14`), so test beyond 16 only after the non-overlap arm succeeds.

**What indicates representation quality rather than rollout exposure is the problem?**
Failure of frozen-latent health/identity gates (low rank/std, high cross-video cosine, tiny or
uninformative target displacement), poor fixed-bottleneck reconstruction/slot alignment, or equal
failure on teacher-forced and rollout states points upstream. Conversely, excellent teacher-forced
fit plus treatment-resistant generated-state failure with healthy representation narrows the issue
to vector-field/conditioning/capacity rather than generic B quality; it does not automatically prove
representation quality is sufficient for future semantics.

## 8. Cost, memory, risks, and observability

### Added calls and resource cost

Control performs one trainable `F_c` call per update. Treatment performs that original call plus
two rollout calls: **two additional calls, three total, a 200% increase in call count and
approximately 3x `F_c` forward work**. Backpropagation traverses all three call graphs, so `F_c`
backward work is also approximately tripled.

The two rollout graphs and generated midpoint must coexist with the original teacher-forced graph
until backward. Coarse-flow activation memory is therefore roughly 3x as a first-order estimate,
plus a small amount for midpoint/displacements. Parameters, optimizer state, frozen encoder
activations, fixed bottleneck state, input batches, and checkpoint size do not triple. Total step
time and peak memory will be less than a naive whole-program 3x in components with substantial
fixed encoder cost, but exact values are **unknown without resource profiling**. The prior plan
records ~23.9 GiB historical allocation and warns that naive scaling could exceed 60 GiB; neither
number certifies the current treatment.

### Failure modes and whether current metrics distinguish them

| Failure mode | Observable signature | Instrumentation sufficient? |
|---|---|---|
| Rollout-loss domination | `loss/rollout * lambda_eff` large relative to inferred total contribution, teacher-forced error >20% worse, rollout may improve while on-path fit degrades | Mostly. Weighted term is derivable, but no direct per-loss gradient norm is logged. Cannot prove domination at gradient level. |
| Unstable rollout gradients | Rising `grad_norm`, `agc_Fc_clipped/max_ratio`, warnings/skips/NaNs; loss/ratios degrade | Yes for outcome/stability, not for which of the two calls/parameters caused it. |
| Shortcut copying/vanishing displacement | Norm ratio toward 0, copy ratio near 1, possibly deceptively small raw displacement | Yes; norm ratio and copy ratio are sufficient when target-valid. |
| Wrong-way or overscaled motion | Low/negative cosine and norm ratio far from 1 | Yes at aggregate level; no per-example distribution or orthogonal/radial decomposition. |
| Poor conditioning use | Low/vanishing correct-vs-shuffled degradation, similar zero/shuffled predictions | Partly. Existing fixed diagnostics support it, but the focused train-rollout schema has no rollout zero/shuffled variants and no attention-level probe. |
| K=2 grid overfit | Training K=2 improves, offline/fixed 4/8-step rollout does not | Only if the unchanged fixed evaluator is run at checkpoints; focused W&B training metrics alone are insufficient. |
| Representation/collapse defect | `c_std_mean`, effective rank, cross-video cosine, frozen hashes/readouts move | Yes for gross failure; these should be invariant in fixed-B mode. |
| Optimization collapse like prior 5350--5500 | Gradient/AGC event followed by condition-shuffle loss, vanishing displacement and ratio failure | Sufficient to identify recurrence/correlation, insufficient to establish causal parameter damage. |
| Future leakage | Suspiciously excellent rollout without generated-state dependence | Code audit and unit contracts are stronger than metrics; current code passes the structural audit. |

Instrumentation is sufficient for the main control/treatment decision **only if** checkpoints 2500
and 5000 also receive matched fixed-batch 1/2/4/8-step evaluation. The current W&B cleanup explicitly
does not forward dynamic multi-K normal/zero/shuffled keys, so training W&B alone cannot apply all
of the original plan's 4/8-step and conditioning gates. No current metric reports per-call gradient
contribution or generated-versus-teacher-forced state distance.

## 9. Final experiment card

| Field | Locked card |
|---|---|
| Control | Fixed present-source `fc_only`; `lambda_rollout=0`; no extra rollout calls. |
| Treatment | Identical arm; `lambda_rollout=0.10`; `rollout_ramp_steps=1500`; K=2 Euler training endpoint loss. |
| Fixed source/target | Same exact Run-60 saved online bottleneck checkpoint/hash for present and future; frozen DINOv3 encoder and B/B_EMA/D. |
| Variables held constant | Current commit, fresh identical `F_c` initialization hash, EGO4D inventory/order, seed 42, batch 64 if both fit, horizon 16, stride 2, `N_c=64`, `D_c=512`, flow architecture/final LayerNorm, dropout 0, bf16, AdamW/betas/decay/LR schedule, AGC/global clipping, teacher-forced loss, evaluation batch/RNG/solver grids, inactive representation/reconstruction losses. |
| Duration/checkpoints | 5,000 updates; evaluate saved states at 2,500 and 5,000. |
| Primary metric | Matched fixed-batch genuine 4/8-step rollout endpoint/copy ratio, not teacher-forced `L_flow` alone. |
| Secondary metrics | Endpoint/batch-mean ratio, K=2 training rollout ratio/loss, teacher-forced loss, displacement cosine/norm ratio, condition-shuffle degradation, gradient norm, AGC, skips/NaNs/warnings, frozen representation/hash checks. |
| Success | Treatment beats control and copy/batch mean on 4/8-step rollouts at both checkpoints, retains condition use and stable gradients, and does not materially sacrifice teacher-forced fit; use the explicit provisional thresholds in §6. |
| Stop | Any NaN/skip/warning, frozen-state change, unrecoverable common-batch resource failure, or sustained treatment-only gradient/AGC collapse. Do not stop solely for one AGC clip. |

### Expected conclusion for each major outcome

| Outcome | Conclusion justified |
|---|---|
| Stable treatment passes all gates at 2500 and 5000 | Strong evidence that generated-state exposure was a major actionable cause of the pre-collapse rollout gap for this fixed latent/horizon. Proceed to robustness, longer rollout grids, then horizon/frame-spacing tests. |
| Treatment improves consistently but remains above copy or misses one gate | Exposure contributes but is insufficient; inspect direction vs magnitude and conditioning before changing representation or architecture. |
| `loss/rollout` falls but fixed 4/8-step metrics do not improve | K=2 objective/grid fit does not generalize; exposure as implemented is not the dominant remedy. |
| Treatment and control improve equally | Null causal result for rollout loss. |
| Early treatment gain, late treatment instability | Useful but unstable gradient tradeoff; investigate weight/ramp and gradient localization. Does not prove the historical 5350 collapse mechanism. |
| Treatment worsens teacher-forced and rollout metrics or conditioning | Objective conflict/domination or unstable gradients; reject this configuration. |
| Both arms fail with unhealthy/informationally weak fixed latents | Upstream representation/horizon/data limitation dominates; rollout exposure is not a fair isolated cure. |
| Both arms have healthy latents and teacher-forced fit, but all rollout interventions fail | Focus next on vector-field architecture, conditioning, multimodality/capacity, or generated-state support—not a generic claim of human-like understanding. |

## 10. Audit limitations

- No training, checkpoint evaluation, dataset access, SSH, or expensive shape smoke test was run.
- The targeted test command `pytest -q tests/test_rollout_loss.py` could not execute because
  `pytest` is not installed in this environment. The test source was inspected, but its pass status
  is therefore unverified here.
- Previous-run numbers are repository-recorded W&B/forensic evidence, not newly fetched live data.
- Exact compute, memory, gradient-component balance, and experimental outcomes remain unknown until
  matched runs and checkpoint evaluations exist.
