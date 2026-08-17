# Pre-Collapse Rollout-Gap Intervention Plan

**Date:** 2026-08-13  
**Scope:** step-2500/5000 present-source behavior only. The later step-5350--5450 instability is
explicitly out of scope. No training/model code, checkpoint, dataset, or run was modified, and no
training was launched.  
**Evidence labels:** **Measured** = raw unsmoothed W&B or executed synthetic test; **Code** = current
executed repository path; **Derived** = mathematical consequence; **Proposed** = not implemented.

## Phase 0: flow-matching mechanism reconstruction

The eight sections below reconstruct the two coarse-flow mechanisms before interpreting the
step-5000 rollout gap. They distinguish the historical Gaussian runs from the present-source run,
and distinguish logical roles such as “target” from the exact module instance that produced each
tensor. Line references are to current commit `df5d959b76038c451ab74218bfa1050c996af6e5` unless a
historical commit is named explicitly.

## 1. Flow internals executive summary

Both variants train the same `CoarseFlow` transformer and the same straight-line rectified-flow
objective. The sole mathematical change is the source endpoint:

```text
Gaussian source:  x0 = epsilon,       x1 = c_future, x_tau = (1-tau)epsilon + tau c_future
Present source:   x0 = c_present,     x1 = c_future, x_tau = (1-tau)c_present + tau c_future
Both:             u = x1 - x0,        L_flow = mean((F_c(x_tau,tau,condition)-u)^2)
```

`losses.interpolate` broadcasts one scalar `tau` per example from `(B,)` to `(B,1,1)` and performs
the affine interpolation; `velocity_target` performs the endpoint subtraction
(`losses.py:31-35,53-80`). Consequently, within an example, every coarse slot and feature uses the
same time, but its endpoint values and velocity remain element-specific.

The source option is **current-code functionality introduced by commit `df5d959`**. Earlier runs
`3y2hxj5t`, `8r6akjsx`, and `r0s6ouwd` predate the `flow_source` configuration key and hardwired
Gaussian construction. Their W&B configs therefore omit that key; absence means the historical
Gaussian implementation, not an unknown setting. The present run `t0okr9cb` records
`flow_source=present` at `df5d959`.

The shape suggested in the investigation prompt, approximately `(64,32,256)`, matches the current
default model configuration when batch size is 64, but **not the named runs**. Their recorded
configs use `B=64`, `N_c=64`, `D_c=512`, so their coarse source, target, interpolant, condition, and
velocity tensors are `(64,64,512)`. Source construction is shape-preserving in both variants.

## 2. Noise tensor trace

### 2.1 Historical full-latent Gaussian path

For full-latent run `8r6akjsx` (commit `7649f8e`), one training example is constructed as follows:

1. `SSV2Dataset._window_indices` chooses an eight-frame context window and an eight-frame future
   window whose end is displaced by `horizon_k`; train starts are randomly sampled, while
   validation starts are centered (`data.py:303-348`). The paired frames receive one shared resize,
   crop, and color-jitter operation before being split (`data.py:350-377`). Collation yields raw
   video tensors `(B,8,3,256,256)` (`data.py:380-395`).
2. The frozen encoder produces detailed present/future features. The online bottleneck produces
   `c_t`; the EMA `TargetBottleneck` produces detached `c_+` (`train.py:499-537`;
   `models.py:358-407`). In the named run, `c_t,c_+` are `(64,64,512)`.
3. Historical `train.py@7649f8e:441-464` sets `flow_target=c_+`, draws
   `epsilon=torch.randn_like(c_+)`, draws `tau=torch.rand(B)`, constructs `x_tau`, predicts the
   velocity, and evaluates MSE.

Thus `epsilon` is sampled from independent standard-normal elements with the exact shape, device,
and dtype of `c_+`: fresh values for each example, coarse slot, and feature coordinate on every
training step. It is not one noise vector shared across the batch or slots and is not normalized or
renormalized after sampling. For the full-latent path there is no scale factor: `epsilon ~ N(0,I)`.

The residual runs `3y2hxj5t` and `r0s6ouwd` are a distinct Gaussian variant. They set the target to
`Delta=c_+ - c_t^EMA` and scale the same-shaped Gaussian by a batch-derived residual standard
deviation (`train.py@7649f8e:449-459`). Their source is therefore
`epsilon_Delta = residual_sigma * randn_like(Delta)`, not standard-normal full-latent noise. This
report does not conflate that coordinate system with `8r6akjsx` or `t0okr9cb`.

### 2.2 Noise, time, condition, and target interaction

For the full-latent objective,

```text
epsilon[b,n,d] ~ N(0,1)
tau[b] ~ Uniform[0,1]
x_tau[b,n,d] = (1-tau[b]) epsilon[b,n,d] + tau[b] c_+[b,n,d]
u[b,n,d] = c_+[b,n,d] - epsilon[b,n,d]
u_hat = F_c(x_tau, tau, c_t)
```

The future coefficient in the teacher-forced input is exactly `tau`: 0% at `tau=0`, 25% at
`tau=.25`, 50% at `.5`, 90% at `.9`, and 100% at `tau=1`. Therefore low-time training states are
noise-dominated, middle states mix source and true future equally, and high-time states directly
reveal most of the future target. `c_t` remains a separate condition-token stream at every time.

Training stochasticity includes the per-element Gaussian, per-example uniform time, randomly
selected/cropped/augmented training clips, minibatch order, condition dropout, and ordinary
optimizer/runtime nondeterminism. In the historical runs, condition dropout was `0.1`: for a
dropped example the complete condition tensor is replaced with learned null slots, while the
Gaussian state and target are unchanged (`models.py:541-548`).

### 2.3 What Gaussian inference would require

A genuine Gaussian rollout samples **one initial** `epsilon` tensor per requested trajectory, then
keeps that initial draw fixed while integrating the deterministic learned ODE:

```text
x^(0)=epsilon
x^(i+1)=x^(i)+Delta_tau F_c(x^(i),tau_i,c_t)
```

Fresh noise is not injected at every Euler step. Different initial draws can yield different
endpoints, so this parameterization has a route to multiple futures; whether the trained model
actually uses that route is empirical and was not established by the historical diagnostics.
Those diagnostics evaluated random teacher-forced line points and one-step endpoint estimates, not
a production Gaussian sampler. The current generic `flow_euler_rollouts` can integrate any supplied
source (`diagnostics.py:488-562`), but the current fixed-arm call passes `abstract` as the source
unconditionally (`train.py:1843-1852`), so it is valid for the present arm and would be wrong for a
fixed Gaussian arm unless the sampled noise were passed instead.

## 3. Present tensor trace

### 3.1 Conceptual present-to-future path

The intended straight line is

```text
x0 = c_t^target
x1 = c_+^target
x_tau = (1-tau)c_t^target + tau c_+^target
u = c_+^target - c_t^target
u_hat = F_c(x_tau, tau, c_t^online)
```

This notation is useful for separating endpoint and conditioning roles, but it is **not the exact
module provenance of `t0okr9cb`**.

### 3.2 Exact `t0okr9cb` implementation

The present run supplied a Run-60 bottleneck checkpoint and entered `_fixed_flow_forward`. That
function sets the loaded bottleneck to evaluation mode and, under one `torch.no_grad()` block,
computes both `present=fixed_bottleneck(e_t)` and `future=fixed_bottleneck(e_+)`
(`train.py:578-595`). `_fixed_flow_training_inputs` then chooses `source=present` because
`flow_source=="present"` (`train.py:624-633`). Finally `train_step` uses the same `abstract`
variable as the explicit condition (`train.py:721-753`). Therefore, for `t0okr9cb`:

```text
p = B_Run60,frozen(E(context))       # no grad
f = B_Run60,frozen(E(future))        # no grad
x_tau = (1-tau)p + tau f
u = f - p
u_hat = F_c(x_tau, tau, p)
```

The source and condition are not merely related: before role/slot embeddings they are numerically
the same tensor `p`. There was no live online-versus-EMA bottleneck distinction in this fixed arm,
and neither endpoint moved during the run. This is stronger and more precise than calling them
“EMA present/future” tensors.

The paired video and encoder trace is otherwise the same as Section 2. For `t0okr9cb`, the W&B
configuration records batch 64, 64 slots, width 512, and horizon 16; hence `p`, `f`, `x_tau`, `u`,
and the explicit condition all have shape `(64,64,512)`.

### 3.3 What “noiseless” does and does not mean

No Gaussian tensor is sampled for the present endpoint. `_fixed_flow_randomness` requests noise
only when `flow_source=="noise"`; otherwise it returns `None` after drawing matched time/dropout
randomness (`train.py:598-633`). For the run's `condition_dropout=0`, the dropout draw is inert.

The run is nevertheless stochastic during training: `tau` is freshly sampled per example and
step, train clips have random start/crop/jitter, minibatch order varies, and optimization/runtime
sources remain. “Noiseless present flow” means deterministic endpoint construction for a fixed
paired clip and fixed bottleneck; it does not mean deterministic training. At inference, a fixed
present clip, fixed preprocessing, fixed weights, and a specified Euler grid produce one
deterministic future. There is no latent random variable left to sample, so the mechanism does not
natively generate multiple futures.

At `tau=0`, the state and condition both contain `p`; after additive stream-role embeddings they
are separate token streams containing the same latent information. At teacher-forced `tau>0`, the
state contains the exact fraction `tau` of the true future. During genuine rollout, only the initial
state is real present; later states contain accumulated model predictions, never ground-truth
future information.

## 4. `F_c` architecture

`CoarseFlow` is unchanged between the historical Gaussian implementation and the present-source
commit; source selection occurs outside it. Its current forward path is:

```text
z_stream = x_tau + slot_pos + z_type
c_stream = condition + slot_pos + cond_type
h0 = concat([z_stream, c_stream], token_dimension)       # (B,2*N_c,D_c)
t = MLP(sinusoidal_embedding(tau))                       # (B,D_c)
hL = six AdaLN-Zero transformer blocks by named-run config
u_hat = affine_LayerNorm(hL[:, :N_c])                    # (B,N_c,D_c)
```

There are **no learned input projection matrices** for the flow-state or condition tensors and no
separate linear output head. Fusion is ordered concatenation `[flow tokens, condition tokens]`
after learned additive slot/role embeddings (`models.py:493-560`). The configured `f_c_dim` is not
used by `CoarseFlow`; the network width is `d_c`.

Each `AdaLNBlock` uses affine-free LayerNorm, full unmasked multi-head self-attention with
`Q=K=V=h`, then an MLP. A time-conditioned modulation MLP produces shift, scale, and residual gate
for both branches; its final linear layer is zero-initialized, so each block starts near identity
(`models.py:410-459`). Attention is bidirectional across all `2*N_c` tokens: state tokens can read
all condition tokens, condition tokens can read all state tokens, and both streams can mix across
slots. Shared `slot_pos` helps bind corresponding slot identities; `z_type` and `cond_type` mark
stream roles. The returned velocity is only the first `N_c` tokens after the final affine
LayerNorm.

Condition dropout is applied before slot/role additions. A dropped example receives the learned
`null_condition` tensor, still stamped with condition slot/type embeddings
(`models.py:541-555`). At `condition_dropout=0.1`, the explicit present condition is absent for an
expected 10% of training examples. At `0`, as in `t0okr9cb`, it is always present.

The final affine LayerNorm forces each output token's pre-affine normalized coordinates to zero
mean and unit variance, followed by learned per-coordinate scale and bias. It therefore constrains
the velocity parameterization and couples feature coordinates, although it does not itself prove
the observed rollout failure.

## 5. Training-versus-inference comparison

### Gaussian source

- **Training:** every example supplies a fresh straight line from independent `epsilon` to its true
  future. At sampled `tau`, the state contains `tau` of the exact future; the label is the constant
  paired displacement `c_+-epsilon`.
- **Inference:** sample `epsilon` once and integrate from it. At `tau=0` the state contains no video
  information; all present information must enter through condition tokens. Subsequent states are
  generated, not teacher-forced.
- **Mismatch:** except at the initial source, training queries lie on exact source-to-target lines,
  whereas inference queries lie on the learned trajectory. There is also a conditional-generation
  burden: the solver starts from an unobserved Gaussian state rather than the observed present.

### Present source

- **Training:** every example supplies the exact line from frozen present to paired future. At
  sampled `tau`, the flow-state already contains `(1-tau)` present plus `tau` true future, while the
  same present is also supplied as the explicit condition.
- **Inference:** start from the exact present and integrate. The initial state is in-distribution,
  but after the first imperfect step the model consumes generated states it never saw in the
  straight-line loss.
- **Mismatch:** source mismatch is removed, but teacher-forced/off-trajectory exposure remains. The
  future shortcut becomes strongest at high `tau`; the model may fit target-revealing states
  without learning a robust field on its own trajectory.

Both objectives expose ground-truth future information in `x_tau` whenever `tau>0`; neither exposes
it during genuine rollout. Present source is more aligned with deterministic transition prediction,
but it does not by itself solve compounding vector-field error. Gaussian source has the larger
overall conceptual mismatch for this observed-present task because it adds an artificial initial
state on top of the shared exposure gap. Conversely, Gaussian conditioning is more structurally
indispensable at `tau=0`: it is the only present-dependent input. In present flow, the explicit
condition duplicates information already present in the state at `tau=0`, although it can remain a
stable anchor as the evolving state moves away from present.

## 6. Gradient comparison

For the historical joint Gaussian full-latent path, `c_+` is detached by `TargetBottleneck` and
`epsilon`, `tau`, and `u` have no learnable ancestry. `L_flow` therefore sends:

- direct gradients through the final affine LayerNorm, every AdaLN block, attention/MLP/time
  parameters, and learned slot/type embeddings of `F_c`;
- indirect gradients into the online bottleneck `B` through the condition `c_t`, except on examples
  replaced by `null_condition`;
- no backpropagation into the frozen encoder or EMA target bottleneck. The EMA bottleneck changes
  only through the post-optimizer moving-average update (`train.py:889-893`).

There are no input “projection weights” to receive gradients: the raw state and condition enter at
width `D_c` and only receive additive learned embeddings. Gradients with respect to the state input
exist mathematically, but Gaussian sampling has no upstream trainable producer. Gradients with
respect to the condition do reach online `B` in joint training.

For `t0okr9cb`, `_fixed_flow_forward` computes present, future, and hence the condition inside
`no_grad`; the fixed optimization scope also disables EMA updates. Thus `L_flow` trains **only
`F_c`**. It updates the same `F_c` parameters as above, including both stream-role embeddings and
all cross-stream attention paths, but cannot change the frozen Run-60 bottleneck or encoder. The
MSE label `f-p` is detached. This difference in trainable representation is at least as important as
the endpoint substitution when comparing historical joint runs with the fixed present run.

For a hypothetical non-fixed present arm using EMA endpoints and online `c_t` only as condition,
the gradient pattern would resemble historical joint training: no endpoint gradients, but an
indirect condition gradient into online `B`, plus separate EMA updates. That is not the executed
`t0okr9cb` path.

## 7. Side-by-side mechanism table

| Property | Historical Gaussian full-latent (`8r6akjsx`) | Fixed present-to-future (`t0okr9cb`) |
|---|---|---|
| Source | Fresh per-element `epsilon ~ N(0,I)` | Frozen Run-60 present latent `p` |
| Target | Detached EMA future latent `c_+` | Frozen Run-60 future latent `f` |
| Condition | Online present latent `c_t` | The same frozen tensor `p` used as source |
| Named-run shape | `(64,64,512)` | `(64,64,512)` |
| Interpolant | `(1-tau)epsilon + tau c_+` | `(1-tau)p + tau f` |
| Velocity label | `c_+ - epsilon` | `f - p` |
| Future fraction in train state | Exactly `tau` | Exactly `tau` |
| Training randomness | Noise, tau, data/order, dropout 0.1, optimizer/runtime | Tau, data/order, optimizer/runtime; dropout draw inert at 0 |
| Inference randomness | Initial noise can index trajectories | None inherent; one deterministic trajectory |
| `F_c` architecture | Same concatenated-token AdaLN transformer | Same concatenated-token AdaLN transformer |
| Condition necessity at `tau=0` | Only present-dependent signal | Informationally redundant with source |
| Direct flow gradient into bottleneck | Yes, via online condition when not dropped | No; bottleneck fixed and under `no_grad` |
| Initial inference state | Artificial Gaussian | Observed present latent |
| Shared exposure problem | Generated states replace true line states | Generated states replace true line states |
| Main shortcut risk | High-`tau` target revelation; condition can be ignored | High-`tau` target revelation plus copy/averaging tendency |
| Multimodal capacity | Different initial noise can yield different endpoints | Deterministic MSE path tends toward one conditional estimate |

Residual Gaussian runs are intentionally omitted from this two-column comparison because their
target is `Delta` and their Gaussian is scaled by residual magnitude; treating them as full-latent
noise would erase a real mechanism difference.

## 8. Answers to the ten mechanism questions

1. **What exactly is the noise tensor, and when is it sampled?** In the historical full-latent
   path it is `torch.randn_like(c_+)`: independent standard-normal values for every batch, slot, and
   feature element, freshly drawn each training step, with no normalization. Residual runs multiply
   that tensor by `residual_sigma`. A genuine inference rollout should sample once at initialization,
   not once per Euler substep.
2. **What exactly is the present tensor, and does the same latent serve as source and condition?**
   In `t0okr9cb` it is the frozen Run-60 bottleneck output on the context clip. Yes: the same
   `abstract` tensor is both source and explicit condition before role embeddings. It is not a live
   online tensor paired with a separate EMA source.
3. **How much future information is present at low, medium, and high `tau`?** Exactly the coefficient
   `tau`: little at low time, half at `.5`, almost all at high time, and the complete future at 1.
   This is true for both variants; genuine rollout contains no direct future at any substep.
4. **Which variant has the larger training-inference mismatch?** Both have straight-line
   teacher-forcing versus generated-state exposure. Gaussian flow additionally begins inference
   from an artificial unobserved source, so it has the larger conceptual mismatch for predicting a
   future from an observed present. The measured size of either gap remains run-specific.
5. **Which variant makes conditioning more structurally useful?** Gaussian flow at `tau=0`, because
   condition tokens are the only carrier of present-video information. Present flow makes the
   condition redundant with the state initially, although the fixed condition can anchor later
   generated states. “Structurally useful” does not establish that attention actually uses it.
6. **Which variant is more prone to copying the present?** Present flow: it starts exactly at
   present and a weak or conditional-mean displacement leaves it near the copy baseline. Gaussian
   flow cannot literally copy by doing nothing because zero displacement leaves Gaussian noise,
   though it can ignore the condition and predict generic futures.
7. **Which is more prone to averaging over futures?** The deterministic present-to-future MSE
   formulation, because one present maps to one predicted velocity field with no sampled latent.
   Gaussian initialization provides a possible multimodal index, but MSE and poor conditioning can
   still collapse it to generic/average behavior; diversity is capacity, not proof.
8. **Which depends most on slot alignment?** Present flow: its label is the slotwise subtraction
   `f[n]-p[n]`, so semantic correspondence of slot `n` across time is essential. Gaussian
   full-latent flow still benefits from condition/target slot alignment, but its source noise has no
   semantic present slot to subtract.
9. **Which implementation details are historical versus current?** Commits through `7649f8e`
   hardwire Gaussian source construction. Commit `df5d959` adds `flow_source`, fixed-bottleneck
   source/target helpers, isolated RNG, and Euler rollout metrics while retaining Gaussian as the
   default. The current checkout contains both source choices; `t0okr9cb` is the present-source run
   at `df5d959`.
10. **What changes mathematically and architecturally between variants?** Mathematically only `x0`
    changes from `epsilon` to `p`, which changes both `x_tau` and the label from `f-epsilon` to
    `f-p`. Architecturally nothing inside `F_c` changes: both use identical token concatenation,
    time conditioning, bidirectional self-attention, AdaLN-Zero blocks, and final LayerNorm. In the
    named experiments the surrounding training scope also changes—historical joint flow could
    shape online `B` through its condition, whereas the fixed present run trains only `F_c`.

## 9. Executive verdict

The step-5000 teacher-forced/rollout gap is **not primarily an inference implementation or solver
accuracy bug**. The actual Euler function integrates an exact constant field to numerical precision,
and the learned checkpoint becomes slightly worse--not better--as Euler resolution increases from
one to eight evaluations. The failure is already present in the first predicted displacement and is
then modestly amplified as the model consumes generated states.

The evidence supports a combination of:

1. **weak/incorrect initial direction:** at step 5000 the one-evaluation predicted displacement has
   cosine only `0.2032` with the true displacement;
2. **magnitude underprediction:** its norm is `102.53` versus `181.84` (`0.564x`);
3. **exposure/vector-field error:** teacher-forced endpoint/copy ratio is `0.0884`, but genuine
   rollout is `1.0445`; further generated-state evaluations increase error to `1.0710` and reduce
   direction cosine to `0.1948`;
4. **declining useful condition effect:** correct-versus-shuffled endpoint advantage shrinks from
   `0.4608` at one evaluation to `0.1697` at eight.

The exact first divergent *substep* and matched-state distance cannot be recovered from W&B scalar
endpoints. The step-2500/5000 weights and intermediate states are absent locally. The strongest
defensible statement is therefore: **the one-evaluation endpoint already fails, and extra
evaluations monotonically amplify rather than repair that failure.**

The selected first intervention is a **two-step differentiable rollout endpoint loss**, added to
the existing random-tau flow loss:

```text
x_hat_0 = x0
x_hat_1/2 = x_hat_0 + 0.5 F(x_hat_0, 0, c_t)
x_hat_1 = x_hat_1/2 + 0.5 F(x_hat_1/2, 0.5, c_t)
L_rollout = MSE(x_hat_1, x1)
L = L_flow + lambda_rollout * ramp(step) * L_rollout
```

This is the smallest mathematically clean experiment that actually presents an on-policy generated
state to F_c and optimizes the deployed endpoint. It requires no invented velocity label away from
the training line. Gradients should pass through both solver steps; the existing teacher-forced
flow loss continues to constrain the intermediate vector field. Use `rollout_train_steps=2`,
`lambda_rollout=0.10`, and a 1,500-step linear ramp as provisional first-arm values. The control is
identical with `lambda_rollout=0`.

## 10. Evidence inherited from the forensic report

The prior report's verified starting points are:

- `t0okr9cb` used one frozen Run-60 bottleneck for both present source and future target,
  `condition_dropout=0`, `N_c=64`, `D_c=512`, and horizon 16.
- Training states were exact paired lines
  `x_tau=(1-tau)x0+tau*x1`, with constant target `u=x1-x0`.
- Future target information enters every teacher-forced state with `tau>0`.
- Genuine rollout starts at present, reuses the same present condition, and feeds each generated
  state into the next call.
- At step 5000, teacher-forced endpoint/copy ratio was `0.08842`; genuine rollout ratios were
  `1.04453--1.07099`.
- The final LayerNorm is restrictive but not established as the primary pre-collapse cause.
- Exact intermediate-state/off-line geometry was not logged and cannot be reconstructed without
  the matching checkpoint and fixed batch.

This plan does not revisit the later collapse and does not use step 10000 to choose an intervention.

## 11. Step-5000 rollout trajectory

### 11.1 Available endpoint sequence

W&B logged completed rollouts for four independent Euler grids, not every intermediate state within
one rollout. The fixed baselines are copy MSE `1.154737`, batch-mean MSE `1.244577`, and true
displacement norm `181.83777`.

| Checkpoint | Euler evaluations | Evaluation times | Endpoint MSE | Endpoint/copy | Predicted displacement norm | Direction cosine |
|---:|---:|---|---:|---:|---:|---:|
| 2500 | 1 | `0` | 1.299510 | 1.125373 | 138.836 | 0.25979 |
| 2500 | 2 | `0, 0.5` | 1.282926 | 1.111011 | 137.021 | 0.26754 |
| 2500 | 4 | `0, .25, .5, .75` | 1.304987 | 1.130116 | 138.213 | 0.26115 |
| 2500 | 8 | `0, ..., .875` | 1.337529 | 1.158297 | 140.178 | 0.25065 |
| 5000 | 1 | `0` | 1.206157 | 1.044529 | 102.529 | 0.20324 |
| 5000 | 2 | `0, 0.5` | 1.211484 | 1.049142 | 103.849 | 0.20305 |
| 5000 | 4 | `0, .25, .5, .75` | 1.220277 | 1.056757 | 105.227 | 0.20079 |
| 5000 | 8 | `0, ..., .875` | 1.236712 | 1.070989 | 107.166 | 0.19484 |

The step-2500 two-evaluation result is slightly better than one evaluation, but remains 11.1% worse
than copy; four/eight reverse the gain. At step 5000, error increases monotonically with solver
resolution. This is inconsistent with truncation error being the dominant problem.

### 11.2 Condition usefulness along completed rollouts

| Step 5000 evaluations | Correct MSE | Shuffled MSE | Zero MSE | Correct advantage over shuffled | Correct advantage over zero |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.206157 | 1.666912 | 1.942138 | 0.460755 | 0.735981 |
| 2 | 1.211484 | 1.458730 | 1.730732 | 0.247246 | 0.519248 |
| 4 | 1.220277 | 1.411984 | 1.636616 | 0.191707 | 0.416339 |
| 8 | 1.236712 | 1.406447 | 1.604369 | 0.169736 | 0.367657 |

The correct condition remains useful, so “condition ignored” is false at step 5000. Its useful
advantage falls as more generated-state calls are made. This is compatible with the predictor
moving into regions where the learned condition/state interaction is less effective. It does not
prove attention-level condition reliance falls; that requires checkpoint interventions.

### 11.3 What can and cannot be concluded

- **First measurable failure:** the `K=1` endpoint, produced solely by `F(x0,0,c_t)`, is already
  worse than copy. Therefore the initial velocity is insufficient before accumulated exposure can
  be the only cause.
- **First divergent substep:** unresolved. A `K=1` full-size update is not the first substep of the
  `K=8` trajectory, and W&B did not save `x_hat_0.125` or other intermediate tensors.
- **Growth pattern:** completed-grid error is gradual/monotonic at step 5000, not an abrupt
  solver-count transition.
- **Dominant error:** direction is the clearest failure (`cos~0.20`); magnitude is also low
  (`0.56--0.59x`). Aggregate means do not permit an exact orthogonal/radial MSE decomposition, so
  “direction dominates” remains high-confidence qualitative rather than exact percentage attribution.
- **Local correction:** the learned field is not usefully corrective over the sampled rollout;
  added evaluations increase displacement slightly in a mostly wrong direction and worsen endpoint
  error. Exact perturbation Jacobians are unavailable.

## 12. Inference implementation audit

The executed rollout is `diagnostics.flow_euler_rollouts` (`diagnostics.py:488-562`).

| Check | Finding | Verdict |
|---|---|---|
| Direction/time interval | forward from `tau=0` toward `1` | Correct |
| Evaluation times | left endpoints `i/K`; final velocity at `1-1/K` | Correct explicit Euler |
| Update sign | `state += dt * velocity` | Correct |
| Step size | exactly `1/K` | Correct |
| Returned endpoint | state after all K updates | Correct |
| Time convention | same scalar tau convention as training and `_timestep_embedding` | Correct |
| Tau shape/dtype | `(B,)`, model time embedding; interpolation broadcasts to `(B,1,1)` | Correct |
| Condition | same tensor reused at every step | Correct |
| Endpoint coordinates | fixed mode uses the same saved online bottleneck for present/future | Correct |
| Dropout | explicit all-false `condition_drop`; configured probability was zero | Correct |
| Train/eval mode | diagnostics do not call `F_c.eval()`, but F_c has no stochastic dropout except the explicitly overridden condition drop; MHA default dropout is zero | No numerical effect in current architecture; hygiene improvement only |
| LayerNorm | applied inside every F_c call exactly as during training | Consistent |
| Autocast | diagnostic flow is outside training autocast; teacher-forced and rollout diagnostics share that evaluation regime | Cannot explain their within-diagnostic gap; a parity check is still advisable |
| Detach/in-place | inference is under `no_grad`; each condition branch starts from `source.clone()`; update is out-of-place assignment | Correct |
| Metric space | endpoint, target, copy, and batch mean share fixed bottleneck coordinates and raw latent MSE | Correct |

One separate diagnostic issue identified in the forensic report concerns the fixed **Gaussian** arm:
the caller currently passes `abstract` as rollout source regardless of `flow_source`. It does not
affect this present-source run.

## 13. Synthetic solver verification

A condition-independent module returning the exact constant tensor `x1-x0` was passed through the
actual `flow_euler_rollouts` function for `K={1,2,4,8}`.

| dtype | K=1 MSE | K=2 MSE | K=4 MSE | K=8 MSE |
|---|---:|---:|---:|---:|
| fp32 | `2.17e-15` | `3.04e-15` | `6.77e-15` | `1.76e-14` |
| bf16 | `7.87e-6` | `1.06e-5` | `2.01e-5` | `6.58e-5` |

The tiny bf16 accumulation error is five orders of magnitude below the learned endpoint MSE. The
solver uses the correct sign, interval, step size, endpoint timing, and repeated condition contract.
The step-5000 failure is not explained by an Euler coding defect.

## 14. Mathematical diagnosis

### 14.1 On-path field

Let `u=x1-x0` and `x_tau=x0+tau*u`. The current target is constant:

```text
dx/dtau = u,       x(0)=x0,
```

whose unique solution is the straight training line and endpoint `x(1)=x1`. On that line,
`u=(x_tau-x0)/tau` for `tau>0`; the state contains increasing target information. The learned
network is constrained only on samples of this paired line, not in a neighborhood around it.

### 14.2 Constant target away from the line

For an off-path state `x_hat_tau=x_tau+delta`, retaining target `u` yields

```text
x_hat_1 = x_hat_tau + (1-tau)u = x1 + delta.
```

It transports the error unchanged. Thus `u=x1-x0` is a valid extension of the original constant
vector field, but it is **neutral, not recovering**. Training noisy states against the same `u`
teaches local invariance of velocity; it does not teach the field to remove displacement error.

### 14.3 Endpoint-correcting field

The natural straight-line recovery from arbitrary state `(x,tau)` to known endpoint `x1` is

```text
v_correct(x,tau) = (x1-x)/(1-tau).
```

It agrees with the original target on the exact path because `x1-x_tau=(1-tau)u`. Solving

```text
dx/dtau = (x1-x)/(1-tau)
```

gives `x1-x(t)=C(1-t)`, so every finite state at `tau<1` reaches `x1` as `tau->1`. For an error
`delta`, the target is `u-delta/(1-tau)`: explicitly corrective.

The denominator is singular at `tau=1`. Along an exact solution the numerator contracts at the same
rate and velocity remains finite, but arbitrary/model errors near one can create enormous labels and
gradients. A discrete K-step training scheme can avoid the singular point by supervising only left
endpoints through `tau=1-1/K`, where the minimum denominator is `1/K`. Clipping the denominator
changes the continuous field and introduces bias; it must be declared rather than presented as the
same objective.

### 14.4 Is the future identifiable from condition off path?

Not uniquely in a multimodal dataset. `c_t` and an off-path state do not determine the realized
future without training-time `x1`; MSE learns a conditional central tendency. At step 5000 correct
condition materially improves rollout, so the architecture carries predictive information, but
there is no evidence it identifies the exact future. This ambiguity limits all deterministic
interventions; it is not a reason to add stochasticity before deterministic rollout beats copy.

### 14.5 Failure classification

- **Exposure bias/vector-field geometry:** strongly supported by the teacher-forced/rollout gap and
  worsening with more generated-state calls.
- **Initial low-tau direction:** independently supported by failed `K=1` endpoint and cosine 0.203.
- **Insufficient conditioning:** partial contributor, not total absence; correct condition helps.
- **Numerical integration:** ruled out as primary.
- **Target ambiguity/multimodality:** plausible irreducible component, not measured.

## 15. Candidate-intervention analysis

### A. Perturb teacher-forced states

Use `x_tilde=x_tau+sigma(tau)eta`. Isotropic perturbation should be scaled from measured rollout
residual RMS by tau band, not guessed in latent units. Until checkpoint trajectories exist, a safe
provisional envelope is relative to true displacement, e.g. RMS noise at `1--5%` of
`RMS(x1-x0)`, zero at endpoints using `sigma(tau)=s*sqrt(tau(1-tau))`. This shape avoids corrupting
the exact source/target boundaries.

Keeping target `u` is mathematically consistent with a parallel constant-field tube but not
corrective: it preserves perturbation to the endpoint. Using `(x1-x_tilde)/(1-tau)` teaches recovery
but introduces the near-one singularity. Isotropic noise may poorly match structured, slot-correlated
rollout errors. This is cheap and causally simple, but should follow measurement of real residuals.

### B. On-policy rollout-state velocity training

Generate `x_hat_tau` with the current model, detach it, then train
`F(x_hat_tau,tau,c_t)` against `(x1-x_hat_tau)/(1-tau)`. Detachment avoids backpropagating through
the behavior rollout, reduces memory, and makes the target a supervised correction at the sampled
state. It does not train earlier actions to place the state better. Without detachment, it becomes a
coupled unrolled objective with greater memory and potentially unstable second-order-like feedback
through target/state construction.

The current predictor as behavior policy changes every optimizer step. An EMA F_c would stabilize
state sampling but adds another stateful module, checkpoint contract, and lag hyperparameter. That
is not the smallest first experiment. Current-policy states under `no_grad`, with a ramp and only
left-endpoint times, are the smallest form. The remaining concern is the explicit corrective label's
singularity and deterministic target ambiguity.

### C. Scheduled mixing

`x_mix=alpha*x_tau+(1-alpha)*x_hat_tau` is neither generally on the original paired line nor a state
produced by the deployed solver. Assigning constant `u` is non-corrective; assigning endpoint
correction changes the field. A schedule from `alpha=1` toward zero resembles scheduled sampling but
does not preserve the original flow-matching objective. It adds a schedule without a cleaner target
than B or D and is not recommended first.

### D. Differentiable endpoint rollout loss

Unroll the actual solver from `x0` and penalize `MSE(x_hat_1,x1)`. Gradients should pass through all
steps for the primary treatment: this teaches the first action and later recovery jointly and
requires no arbitrary off-path velocity label. `K=2` is the minimum that exposes F_c to its own
intermediate state; `K=1` is only a low-tau endpoint regression and cannot test exposure recovery.

Endpoint-only loss admits poor/canceling intermediate fields, but retaining the original random-tau
`L_flow` prevents it from being the only constraint. Stop-gradient or truncated backprop saves
memory but changes the causal question and should be a later ablation only if full two-step memory
fails preflight.

### E. Multistep endpoint consistency

Require K=1,2,4 integrations to produce compatible endpoints, optionally with target endpoint loss.
Consistency alone can make several solvers agree on the same wrong/copy endpoint. With target loss
it is useful but costs seven extra F_c calls for 1+2+4 and mixes solver robustness into the first
causal test. It is not cleaner than one K=2 target endpoint loss.

### F. Solver-only changes

More Euler steps already worsen step-5000 performance. Heun/RK2 or RK4 can reduce numerical error
for a smooth accurate field, but cannot make a wrong/off-support field correct and adds evaluations
at generated states. A later evaluation-only Heun comparison is useful once weights exist; current
evidence predicts no material repair. Adaptive integration is unsupported and unjustified.

### G. Stronger structural conditioning

Correct condition substantially beats shuffled and zero at step 5000, proving useful condition
access. The declining gap warrants future attention probes, but cross-attention redesign would
confound the rollout-objective test. Keep architecture fixed first.

### H. Tau reweighting

Oversampling or weighting `tau<=0.25` is the cheapest way to improve the initial direction, and the
failed K=1 result makes it credible. It cannot teach recovery from generated mid-trajectory states.
It is therefore the best low-cost fallback, not the primary structural intervention. A treatment
that improves K=1 but leaves K=4/8 near copy would falsify it as a sufficient solution.

## 16. Ranked interventions

Ranking weights mathematical correctness, directness, complexity/cost, stability, and causal
interpretability.

1. **Two-step differentiable endpoint rollout loss.** Direct target, actual generated state,
   minimal solver length, no invented off-path velocity.
2. **Low-tau reweighting/oversampling.** Cheapest and targets demonstrated initial-direction error,
   but not recovery.
3. **Detached on-policy corrective-velocity supervision.** Direct recovery training and lower
   memory, but changes the target field and must manage `1/(1-tau)`.
4. **Measured tube perturbation with corrective target.** Simple after real residual scale/shape is
   known; currently uncalibrated.
5. **Two-step endpoint loss with truncated gradient.** Lower memory, weaker credit assignment.
6. **Multistep endpoint consistency plus target.** Sound but expensive and less isolated.
7. **Heun/RK2 evaluation.** Diagnostic, unlikely treatment.
8. **Directed cross-attention.** Potential later architecture fix; condition is already useful.
9. **Scheduled mixing.** Ambiguous state distribution/target.
10. **More Euler/RK4/adaptive steps alone.** Existing evidence argues against benefit.

## 17. Selected first experiment

### 17.1 Matched arms

**Control:** existing fixed present-to-future objective, `lambda_rollout=0`.  
**Treatment:** identical objective plus differentiable two-step endpoint loss,
`lambda_rollout=0.10`, `rollout_train_steps=2`, `rollout_warmup_steps=1500`.

Both start fresh from the same exact Run-60 frozen bottleneck checkpoint and identical fresh F_c
initialization hash. They train for **5,000 updates**, not 10,000; checkpoints at 2,500 and 5,000
are the required persistence test. This isolates the pre-collapse regime without investigating the
later event.

`lambda=0.10` is provisional. At historical step 5000, endpoint MSE was about `1.21` while random
flow loss was about `0.1--0.2`; weight 0.1 puts the added scalar in the same order without letting
it dominate. The ramp protects initialization. Do not tune the treatment after seeing its result;
if its gradient ratio is grossly outside a pre-registered safe range in resource preflight, abort
both arms and revise the plan.

### 17.2 Locked controls

Hold fixed: commit, source checkpoint/hash, frozen DINOv3 encoder, frozen one-copy bottleneck policy,
EGO4D inventory/order, seed 42, batch 64, horizon 16, F_c architecture/init, final LayerNorm,
condition dropout 0, AdamW/betas/weight decay, LR `1e-4` and schedule, AGC/global clipping,
teacher-forced loss, evaluation batch, diagnostic RNG, and evaluation Euler K values.

### 17.3 Resource estimate

The control performs one trainable F_c forward/backward per update. Treatment performs that plus a
two-call differentiable rollout: approximately **3x F_c forward/backward work** and retains roughly
**3x F_c activation graphs**. Frozen encoder cost is unchanged, so total wall time should rise by
less than 3x but cannot be estimated precisely without profiling. The historical arm allocated
about `23.9 GiB` on an A100; naive activation scaling could exceed 60 GiB but is not guaranteed
because fixed encoder/module/optimizer memory does not scale. Batch 64 on 80 GiB is plausible, not
certified. An exact one-step resource preflight is mandatory; if it fails, use gradient checkpointing
inside the unroll or a common smaller batch for **both** arms. Do not silently truncate gradients.

## 18. Exact implementation locations

No changes are made now. If approved:

- `config.py` `TrainConfig` near `flow_source` (`config.py:276-284`): add
  `lambda_rollout: float = 0.0`, `rollout_train_steps: int = 2`, and
  `rollout_warmup_steps: int = 1500`. Zero must preserve current behavior and RNG exactly.
- `configs/train.yaml` training-loss section: document the three fields with zero default.
- `train.py:721-754`: after the ordinary F_c flow forward, call a pure differentiable Euler helper
  only when `lambda_rollout>0`; use detached fixed `abstract/target_abstract` endpoints and the same
  explicit present condition; add ramped endpoint MSE to total loss.
- `train.py:894-917`: log `L_rollout`, `rollout_scale`, and optionally
  `rollout_train_endpoint_vs_copy_ratio`; zeros when disabled.
- `train.py:2680-2785,3070-3092`: expose `--lambda-rollout`, `--rollout-train-steps`, and
  `--rollout-warmup-steps` as explicit experiment overrides.
- `diagnostics.py:488-562`: keep evaluation solver semantics unchanged; optionally factor the
  tensor-returning Euler core so train/eval share sign/time/update logic. Evaluation metrics remain
  under `no_grad`.
- `provenance.py`: ensure the resolved new fields enter run identity automatically; add explicit
  common-identity tests if current recursive serialization does not already cover them.
- `tests/test_optimizer_and_flow.py`: add contracts below.
- Agent-maintained docs: update `AGENT_FILES/AGENTS.md`,
  `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`, and `GUIDES/CODEBASE_STRUCTURE.md` only if a new
  evaluator file is added.

## 19. Test and training commands

These commands describe the proposed implementation. **Do not run them before approval.** The new
flags do not exist yet.

### Required unit tests

1. Constant field: differentiable K=2 rollout reaches `x1` and has zero endpoint loss.
2. Zero flag: `lambda_rollout=0` produces byte-identical RNG, loss, gradients, and optimizer update
   to current behavior.
3. Gradient route: treatment gradients reach F_c through both Euler calls and never reach frozen
   encoder/B/B_EMA/D.
4. Generated-state exposure: second F_c call receives the first call's updated state, not the
   teacher-forced midpoint.
5. Time/update contract: calls at `tau={0,0.5}`, `dt=0.5`, endpoint after the second update.
6. Condition contract: same present tensor, explicit no-drop behavior, every call.
7. Provenance/resume: fields serialize, checkpoint, and restore strictly.
8. Finite behavior under bf16 autocast.

```bash
.venv/bin/python -m pytest -q tests/test_optimizer_and_flow.py tests/test_provenance.py
```

### Required smoke/resource preflight

```bash
python train.py --resource-preflight \
  --data ego4d --encoder dinov3_vitb16 \
  --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 --encoder-frame-microbatch 32 \
  --seed 42 --steps 5000 --batch-size 64 --horizon-k 16 \
  --flow-source present \
  --flow-bottleneck-checkpoint /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt \
  --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 \
  --decoder-dim 512 --decoder-blocks 4 --condition-dropout 0 \
  --lambda-var 0 --lambda-cov 0 --lambda-recon 0 --lambda-recon-pred 0 \
  --lambda-rollout 0.10 --rollout-train-steps 2 --rollout-warmup-steps 1500 \
  --lr-coarse-flow 1e-4 \
  --checkpoint-dir /workspace/ckpt/inv023_rollout_endpoint_treatment \
  --provenance-out logs/preflight/inv023_rollout_endpoint_treatment.json
```

### Exact matched training commands

Control:

```bash
python train.py \
  --data ego4d --encoder dinov3_vitb16 \
  --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 --encoder-frame-microbatch 32 \
  --seed 42 --steps 5000 --batch-size 64 --horizon-k 16 \
  --flow-source present \
  --flow-bottleneck-checkpoint /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt \
  --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 \
  --decoder-dim 512 --decoder-blocks 4 --condition-dropout 0 \
  --lambda-var 0 --lambda-cov 0 --lambda-recon 0 --lambda-recon-pred 0 \
  --lambda-rollout 0 --rollout-train-steps 2 --rollout-warmup-steps 1500 \
  --lr-coarse-flow 1e-4 \
  --checkpoint-dir /workspace/ckpt/inv023_rollout_endpoint_control \
  --log-every 50 --diag-every 500 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv023_precollapse_rollout_endpoint \
  --wandb-name "Investigation 23 · Rollout endpoint consistency · Present-flow control" \
  --require-wandb
```

Treatment:

```bash
python train.py \
  --data ego4d --encoder dinov3_vitb16 \
  --encoder-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --encoder-precision bf16 --encoder-frame-microbatch 32 \
  --seed 42 --steps 5000 --batch-size 64 --horizon-k 16 \
  --flow-source present \
  --flow-bottleneck-checkpoint /workspace/ckpt/inv019_bottleneck_shape_covvar/inv019_covvar_dinov3_n64_d512_m512/phase1_step15000.pt \
  --n-c 64 --d-c 512 --bottleneck-mixer-dim 512 \
  --decoder-dim 512 --decoder-blocks 4 --condition-dropout 0 \
  --lambda-var 0 --lambda-cov 0 --lambda-recon 0 --lambda-recon-pred 0 \
  --lambda-rollout 0.10 --rollout-train-steps 2 --rollout-warmup-steps 1500 \
  --lr-coarse-flow 1e-4 \
  --checkpoint-dir /workspace/ckpt/inv023_rollout_endpoint_treatment \
  --log-every 50 --diag-every 500 \
  --wandb-entity smahalanobis-uc-davis --wandb-project hjepa-vwm \
  --wandb-group inv023_precollapse_rollout_endpoint \
  --wandb-name "Investigation 23 · Rollout endpoint consistency · Two-step weight 0.10" \
  --require-wandb
```

### Exact evaluation command

The current repository has no checkpoint CLI that returns intermediate rollout states. If approved,
add a read-only `evaluate_rollout_gap.py` using the shared Euler core, then run:

```bash
python evaluate_rollout_gap.py \
  --checkpoints \
    /workspace/ckpt/inv023_rollout_endpoint_control/phase1_step2500.pt \
    /workspace/ckpt/inv023_rollout_endpoint_control/phase1_step5000.pt \
    /workspace/ckpt/inv023_rollout_endpoint_treatment/phase1_step2500.pt \
    /workspace/ckpt/inv023_rollout_endpoint_treatment/phase1_step5000.pt \
  --steps 1 2 4 8 \
  --output /workspace/forensics/inv023_precollapse_rollout_gap.json
```

The evaluator must be inference-only, verify checkpoint/dataset/encoder identities, use the pinned
fixed source-diverse validation batch, and never save model state.

### Required W&B metrics

- existing teacher-forced and rollout normal/zero/shuffled metrics;
- `L_rollout`, `rollout_scale`, train two-step endpoint/copy ratio;
- per K: endpoint/copy and endpoint/batch-mean ratios, displacement norm/true ratio and direction
  cosine, correct-minus-shuffled/zero gaps;
- evaluation-only per substep: generated-vs-teacher state RMS/cosine/moments, velocity MSE/cosine/
  norm ratio on both, correction cosine toward `x_tau` and `x1`;
- existing `grad_norm`, skips/NaNs/warnings, F_c AGC counts, and frozen bottleneck reconstruction/
  representation metrics.

## 20. Success and falsification gates

Evaluate both steps 2500 and 5000 on identical fixed batches. Required gates are:

1. **Hard:** treatment 4- and 8-step endpoint/copy ratio `<1.0` at both checkpoints.
2. **Hard:** treatment 4- and 8-step endpoint/batch-mean ratio `<1.0` at both checkpoints.
3. **Hard:** treatment improves both ratios over matched control at both checkpoints.
4. **Provisional numerical:** predicted/true displacement norm ratio in `[0.40,1.25]` and at least
   `0.10` absolute above a degenerate near-zero predictor.
5. **Provisional numerical:** displacement-direction cosine `>=0.30` and at least `+0.05` over
   control. Historical step-5000 value is ~0.20.
6. **Provisional numerical:** correct-condition endpoint MSE at least 10% below shuffled during
   4/8-step rollout.
7. **Hard relative:** teacher-forced-to-rollout degradation, defined as
   `rollout_endpoint_mse / teacher_forced_endpoint_mse`, is at least 30% lower than control while
   teacher-forced error does not worsen by more than 20%.
8. **Hard:** gates 1--7 persist at both 2500 and 5000, not one lucky checkpoint.
9. **Hard:** no NaN, skipped update, instability warning, or frozen-state hash change through 5000.
10. **Hard:** fixed bottleneck representation/reconstruction values remain numerically invariant
    within diagnostic precision; any change indicates an implementation/provenance defect.

The exposure/off-path hypothesis is falsified as the dominant actionable cause if the treatment
clearly reduces training two-step endpoint loss and sees generated states, yet genuine 4/8-step
rollout does not improve over control at either checkpoint. It is also weakened if a solver-only
Heun evaluation on the unchanged step-5000 checkpoint beats copy by a material margin, or if future
checkpoint probing shows generated and teacher-forced states/velocity errors remain nearly equal
despite the endpoint gap.

## 21. Evidence limitations

- Step-2500/5000 checkpoint files and exact intermediate rollout states are not local.
- W&B contains endpoints after complete K-step integrations, not every substep. Consequently the
  exact first divergence time, state distance, velocity error on generated states, correction
  Jacobian, and attention-level condition reliance remain unmeasured.
- Logged displacement norms and cosines are means over examples; they do not support an exact
  direction-versus-magnitude MSE decomposition.
- The fixed W&B validation batch used adjacent chunks from one source UID. “Shuffled” is a
  wrong-chunk condition, not verified different-video conditioning.
- The resource estimate is analytical. A full-batch treatment preflight is mandatory.
- Thresholds labelled provisional are mechanism-sensitive starting gates, not established project
  acceptance standards.

## 22. Answers to the ten rollout-gap questions

1. **Is step-5000 rollout failure caused by an inference bug?** No identified bug explains it. The
   actual constant-field solver is correct to fp32/bf16 precision, and learned error worsens with
   more evaluations. A train/eval dtype parity probe is worthwhile but cannot explain the
   teacher-forced/rollout comparison within the same diagnostic regime.
2. **At which integration step does divergence first become significant?** Exact substep unknown
   because intermediate states were not logged. The earliest measurable result--the K=1 endpoint
   from the tau-zero field--is already worse than copy (`1.0445x`).
3. **Direction, magnitude, or conditioning?** Primarily poor direction (cosine ~0.20), with material
   under-magnitude (~0.56x) and declining but nonzero useful conditioning over longer rollouts.
4. **Does increasing solver accuracy fix it?** No. Step-5000 ratios worsen monotonically from
   `1.0445` to `1.0710` as K increases 1 to 8.
5. **Is the vector field locally corrective or error-amplifying?** Over completed rollouts it is
   weakly error-amplifying/non-corrective: more calls raise endpoint error and lower alignment.
   Local perturbation derivatives remain unmeasured.
6. **What future information is missing from generated states?** Teacher-forced `x_tau` contains
   the realized target component `tau*x1` exactly and remains paired to the correct straight-line
   displacement. Generated states contain only earlier model predictions; their errors do not
   encode the missing realized future component.
7. **What is the correct off-manifold training target?** There is no unique extension required by
   the original flow objective. Constant `x1-x0` is neutral to state error. The endpoint-correcting
   `(x1-x)/(1-tau)` is mathematically recovering and agrees on-path, but is singular near one and
   changes off-path geometry. Endpoint loss avoids assigning an arbitrary off-path velocity.
8. **Which rollout-aware intervention is mathematically best justified?** Differentiable target
   endpoint loss through the deployed solver, retained alongside original flow matching.
9. **Which is the smallest clean experiment?** A K=2 differentiable endpoint loss with weight 0.10
   and ramp 1500 versus a zero-weight matched control, stopping at step 5000.
10. **What falsifies exposure/off-manifold as the actionable cause?** The treatment sees generated
    states and reduces its rollout training loss but produces no genuine 4/8-step improvement at
    both checkpoints; or an unchanged-checkpoint solver-only method materially repairs rollout.

| Candidate | Failure targeted | Evidence | Risk | Cost | Rank |
| --------- | ---------------- | -------- | ---- | ---- | ---- |
| Two-step differentiable endpoint loss | Initial action plus generated-state recovery | Directly matches failed deployed endpoint; no off-path label needed | Coupled gradients, memory, possible cancellation | High: ~3x F_c graph work | 1 |
| Low-tau reweighting/oversampling | Weak first direction | K=1 already fails; tau-zero condition is informative | Does not teach recovery | Low | 2 |
| Detached on-policy corrective velocity | Generated-state recovery | Correcting field derived and on-path compatible | `1/(1-tau)` singularity; moving policy | Medium | 3 |
| Perturbed tube plus corrective target | Local robustness/recovery | Plausible narrow-manifold issue | Error scale/shape unmeasured | Low-medium | 4 |
| Truncated endpoint backprop | Generated-state recovery with less memory | Same endpoint target | Weak first-step credit assignment | Medium | 5 |
| Multistep target consistency | Solver/discretization robustness | Several K values fail differently | Can agree on wrong endpoint | Very high | 6 |
| Heun/RK2 only | Numerical truncation | No checkpoint comparison yet | Cannot fix wrong field | Medium eval cost | 7 |
| Directed cross-attention | Condition use off path | Condition advantage shrinks but remains material | Confounds architecture | Medium-high | 8 |
| Scheduled mixing | Exposure | Generic scheduled-sampling intuition | State/target not principled | Medium | 9 |
| More Euler/RK4/adaptive only | Numerical accuracy | More Euler calls already worsen result | More off-support evaluations | Medium-high | 10 |
