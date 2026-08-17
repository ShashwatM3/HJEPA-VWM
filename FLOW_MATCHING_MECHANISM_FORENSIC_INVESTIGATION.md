# Coarse Flow-Matching Mechanism: Forensic Investigation

**Date:** 2026-08-13  
**Scope:** read-only repository, artifact, and raw W&B investigation. No training code,
checkpoint, dataset, or W&B run was modified; no training was launched.  
**Evidence labels:** **Code** = executed path in commit `df5d959`; **Measured** = existing artifact or
raw unsmoothed W&B history; **Derived** = algebraic consequence of code; **Hypothesis** = not yet
causally isolated.

## 1. Executive verdict

The leading explanation is **substantially correct but conflates two systems and two failures**.

1. The repository's historical/default objective is Gaussian `noise -> future`. Its training state
   contains the ground-truth future at every `tau > 0`, while inference must create that state from
   noise. This gives a real teacher-forced shortcut and a real exposure/off-manifold risk.
2. The investigated W&B run `t0okr9cb` was **not** that default system. Its live resolved config
   proves it used a frozen Run-60 bottleneck, `flow_source=present`, `condition_dropout=0`, and
   `(x0,x1)=(c_present,c_future)`. Thus it already began inference exactly from the available
   present latent. Setting Gaussian noise to zero is neither what this run did nor mathematically
   equivalent to present-to-future flow.
3. Nevertheless, present-to-future training still teacher-forces the exact future into
   `x_tau=(1-tau)c_present+tau*c_future` for every `tau>0`. Before the acute collapse, the run had
   excellent random-tau endpoint error but failed genuine Euler rollout: at step 5000,
   teacher-forced endpoint/copy ratio was `0.08842`, whereas one-step and eight-step rollout ratios
   were `1.04453` and `1.07099`. This is direct evidence of an **on-trajectory versus generated-state
   generalization failure**, although raw matched-state distances were not logged and therefore the
   stronger geometric claim “left the interpolation manifold by X” cannot be quantified.
4. A separate acute failure occurred after step 5000. The first raw training-health warning was at
   step 5350 (`grad_norm=15.3487`, six F_c tensors AGC-clipped), the largest excursion was at 5450
   (`grad_norm=38.9573`, eight clipped), and the first diagnostic collapse was at 5500. Correct,
   zero, and shuffled conditions then became indistinguishable and displacement progressively
   vanished. Temporal coincidence is strong; causation is unproven without exact checkpoint replay
   or component transplants.
5. Unrestricted concatenated self-attention and adaLN-Zero make condition use optional and initially
   delayed, but neither alone proves the learned shortcut. Condition dropout cannot explain
   `t0okr9cb` because it was exactly zero. The final affine LayerNorm is a genuine velocity-output
   restriction, but existing magnitude/moment evidence and the pre-collapse condition-sensitive
   model make it a secondary concern, not the dominant established failure.

The clean next experiment is **not another present-source implementation**; that path already
exists and has run. The smallest causal follow-up is a matched, checkpointed replication of the
existing fixed-coordinate `noise` and `present` arms with the current diagnostics extended
evaluation-only to matched-tau teacher-forced/generated states. Because the first present arm
underwent an acute collapse, the primary comparison must use a pre-registered stable window and a
collapse-invalidity rule.

## 2. Implemented training mechanism

### 2.1 Clips and frozen features

`SSV2Dataset._window_indices` (`data.py:303-324`) chooses eight context indices
`s+i*stride` and eight future indices whose corresponding positions are shifted by `horizon_k`.
`__getitem__` (`data.py:350-377`) decodes the combined list, applies one shared resize/crop/color
transform, and returns two raw `[0,1]` tensors of shape `(8,3,256,256)`. Collation
(`data.py:380-395`) produces `(B,8,3,256,256)` context and target clips.

`_coarse_forward` (`train.py:499-537`) calls the frozen encoder on each clip under
`torch.no_grad()`. The selected DINOv3 run contract yields detailed features
`e_t,e_plus: (B,2048,768)`; generic shape is `(B,N_e,D_e)`. `FrozenEncoder` permanently disables
parameter gradients (`encoders.py:203-268`). Future pixels therefore enter training only through a
frozen feature/target path.

### 2.2 Historical/default Gaussian-source path

For the non-fixed default path (`train.py:729-754`):

```text
c_t      = B_online(e_t)                 # attached condition
c_plus   = B_EMA(e_plus)                 # detached future target
epsilon  = randn_like(c_plus)
tau      = Uniform[0,1), one scalar/example
z_tau    = (1-tau) epsilon + tau c_plus
u        = c_plus - epsilon
u_hat    = F_c(z_tau, tau, c_t)
L_flow   = mean((u_hat-u)^2)
```

For `t0okr9cb`, `N_c=64,D_c=512`; repository defaults are `32,256`. `interpolate` broadcasts
`tau: (B,)` to `(B,1,1)` (`losses.py:31-35,53-66`), `velocity_target` subtracts endpoints
(`losses.py:69-80`), and `flow_matching_loss` is unweighted elementwise mean MSE
(`losses.py:109-118`). There is no tau-band reweighting.

At `tau=0`, `z_0=epsilon` contains no future target. At `tau>0`, `z_tau` contains the exact
supervised future with coefficient `tau`. This is target information in a teacher-forced input,
not future information available at inference.

### 2.3 Implemented fixed present-source path

Commit `df5d959` already implements `flow_source=noise|present`. A fixed checkpoint loads only its
saved online bottleneck, copies it exactly into the target wrapper, and freezes both
(`train.py:1507-1518`). `_fixed_flow_forward` applies that **same frozen bottleneck** to present and
future detailed features (`train.py:578-595`). `_fixed_flow_training_inputs`
(`train.py:598-633`) produces matched tau/dropout streams and selects:

```text
x0 = c_present = B_fixed(e_t)
x1 = c_future  = B_fixed(e_plus)
x_tau = (1-tau)x0 + tau*x1
u_tau = x1-x0
u_hat = F_c(x_tau,tau,c_present)
```

All endpoint tensors are under `no_grad`. In this fixed mode `B` and `B_EMA` are identical frozen
copies; F_c trains, and the unused decoder may remain trainable by configuration but does not enter
the loss when reconstruction weights are zero. No EMA update occurs because
`apply_optimization_scope` sets `updates_ema=False` for any fixed checkpoint
(`train.py:174-218,889-893`).

### 2.4 Why `epsilon=0` is not present-to-future flow

With `epsilon=0`, the historical construction is

```text
z_tau=tau*c_future,       u=c_future.
```

Present-to-future is

```text
x_tau=(1-tau)c_present+tau*c_future,
u=c_future-c_present.
```

They agree only in the degenerate case `c_present=0`. Their starting state, velocity target,
trajectory, and inference boundary condition all differ. The implementation correctly selects
`source=present`, not zero noise (`train.py:624-633`).

## 3. Implemented rollout mechanism

`flow_euler_rollouts` (`diagnostics.py:488-562`) is the only implemented coarse rollout evaluator.
For `t0okr9cb`, its source is the fixed present latent (`train.py:1843-1853`). For each function
evaluation count `K in {1,2,4,8}` it performs explicit forward Euler:

```text
state_0 = source
dt = 1/K
for i=0..K-1:
    tau_i = i/K
    state_{i+1} = state_i + dt*F_c(state_i,tau_i,condition)
```

Thus integration is forward on `[0,1]`, evaluates at left endpoints
`{0,1/K,...,(K-1)/K}`, reuses one fixed condition at every call, and feeds each generated state
back to the predictor. There is no projection, renormalization, clipping, stochastic correction,
adaptive solver, classifier-free guidance, or learned dropout consumer. Diagnostics force dropout
off. The returned final state is compared with `c_future` by endpoint MSE/cosine, displacement norm
and alignment. Copy-present and batch-mean-future MSEs are fixed baselines.

Important limitation: the function always receives `abstract` as source in the current fixed-flow
diagnostic call (`train.py:1845-1851`). It therefore evaluates genuine present-start rollout for
the present arm, but **would not evaluate a genuine Gaussian-start rollout for a fixed noise arm**.
That is a concrete diagnostic mismatch in the proposed matched source comparison and must be fixed
in evaluation before interpreting the noise arm.

No general production inference sampler exists. These diagnostic Euler rollouts are the implemented
inference evidence.

## 4. Tensor and gradient trace

| Operation | Code | Input -> output shape | Frames/network | Gradient and consumer |
|---|---|---|---|---|
| Window construction | `data.py:303-377` | video -> two `(8,3,256,256)` clips | present and shifted future | raw data; shared transforms |
| Batch collation | `data.py:380-395` | samples -> `(B,8,3,256,256)` each | both | no autograd yet |
| Present encoder | `train.py:527-531` | context -> `e_t (B,N_e,D_e)` | present, frozen E | `no_grad`; consumed by B and recon target |
| Future encoder | `train.py:532-536` | target -> `e_plus` | future, frozen E | `no_grad`; consumed by target B |
| Online present B | `train.py:531` | `e_t -> c_t (B,N_c,D_c)` | present, online B | attached in historical joint mode; condition, geometry, recon |
| EMA future B | `models.py:397-407` | `e_plus -> c_plus` | future, B_EMA | `no_grad` plus `as_target`; flow target |
| Fixed endpoints | `train.py:578-595` | `e_t,e_plus -> c_present,c_future` | same frozen saved online B | both detached; flow source/target/condition |
| Gaussian source | `train.py:745-747` | shape of target -> epsilon | neither frames nor network | no grad; interpolant and velocity target |
| Tau | `train.py:749`; fixed `598-621` | random `(B,)` | neither | broadcast to `(B,1,1)`; input/interpolant |
| Interpolant | `losses.py:53-66` | endpoints,tau -> `(B,N_c,D_c)` | contains future for tau>0 | attached only through source if source attached; flow input |
| Velocity target | `losses.py:69-80` | target-source -> same | supervised future endpoint | detached endpoints; MSE target |
| Condition dropout | `models.py:541-548` | `c_t -> c_t/null` | present or learned null | per example; gradient to B unless dropped; `t0okr9cb p=0` |
| Token stamping | `models.py:498-516,549-557` | state and condition -> `(B,2N_c,D_c)` | future-bearing state + present | attached to F_c; learned slot/type codes |
| Time embedding | `models.py:462-479,517-519` | `(B,) -> (B,D_c)` | no frames | trains time MLP and block modulation |
| adaLN blocks | `models.py:410-459` | `(B,2N_c,D_c)` -> same | both streams | six zero-init gated residual blocks |
| Attention | `models.py:424-456` | Q=K=V=all `2N_c` tokens | bidirectional | no mask; state and condition can read each other |
| Output | `models.py:523,560` | first `N_c` tokens -> `(B,N_c,D_c)` | predicted velocity | affine LayerNorm; **no Linear head** |
| Flow loss | `train.py:751-754` | prediction,target -> scalar | target future supervision | trains F_c and, in joint mode, B through condition |
| Backward/AGC | `train.py:861-888` | scalar -> parameter grads | trainable modules | AGC, global clip 0.5, then AdamW unless skipped |
| EMA update | `train.py:889-893` | B -> B_EMA | online to target | no grad; only successful non-fixed joint steps |

Stop-gradient boundaries are the frozen encoder `no_grad` blocks, `TargetBottleneck.forward`'s
`no_grad` and `as_target`, and the entire fixed-flow forward. The historical online condition is
deliberately **not detached**, so flow loss can shape B through condition. Future frames never enter
the model as inference-available observations. They enter (a) detached target construction and
(b) the teacher-forced interpolated state during training/evaluation. At inference only the source
(noise historically, present latent in present mode), present condition, time, and previous
generated state are available.

## 5. Training-versus-inference mismatch

### 5.1 Concrete mismatches

| Mismatch | Noise source | Present source |
|---|---|---|
| Exact future mixed into training state for `tau>0` | Yes | Yes |
| Inference start equals an observed latent | No; fresh Gaussian | Yes; exact present latent |
| Intermediate state guaranteed to lie on paired straight line | Training yes; rollout no | Training yes; rollout no |
| Predictor sees its own accumulated error | Training no | Training no |
| Same condition reused throughout rollout | Yes | Yes |
| State normalization/projection during rollout | None | None |
| Euler discretization error | Yes | Yes |
| Objective weights off-trajectory correctness | No | No |

The shortcut can be seen algebraically. On a valid straight path with known source `x0`,
`x_tau=x0+tau*u`, so for `tau>0`, `u=(x_tau-x0)/tau`. In present mode the explicit condition is
also `x0`; the network can reconstruct velocity from the teacher-forced state plus present without
learning a robust vector field away from that line. Near `tau=1`, `x_tau` is already almost the
answer. In Gaussian mode, the state alone increasingly reveals `c_future`; the independent source
noise is not separately supplied, but distributional denoising can still exploit that leakage.

Present-source therefore removes the unknown-Gaussian boundary mismatch and makes the initial state
exactly available. It does **not** remove teacher forcing, intermediate future leakage, self-error
exposure, multimodal averaging, or optional condition use.

## 6. Conditioning-ablation evidence

### 6.1 What is actually measured

Raw unsmoothed W&B history for `t0okr9cb` contains random-tau correct, rolled-batch, and zero
conditions plus genuine Euler normal/rolled/zero conditions. It does not contain explicit tau
bands, velocity cosine/norm ratio, different-source identity, learned-null, attention-blocked, or
direction-error angles. `torch.roll` is not proven different-source for the recorded validation
batch; the run provenance lists 16 adjacent chunks from one source UID. It is therefore a
wrong-chunk condition, not a verified different-video condition.

Representative measured values:

| Step | TF correct MSE | TF rolled MSE | TF zero MSE | TF endpoint/copy | 1-step rollout correct/rolled/zero MSE | 1-step ratio | 8-step ratio | rollout displacement/alignment (1-step) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1500 | 0.4970 | 0.7379 | 0.7815 | 0.4304 | 1.6490 / 1.6824 / 1.9682 | 1.4281 | 1.4376 | 160.09 / 0.164 |
| 3000 | 0.1417 | 0.6298 | 0.6436 | 0.1227 | 1.2631 / 1.5697 / 1.8716 | 1.0939 | 1.1174 | 125.26 / 0.237 |
| 5000 | 0.1021 | 0.5967 | 0.5811 | 0.0884 | 1.2062 / 1.6669 / 1.9421 | 1.0445 | 1.0710 | 102.53 / 0.203 |
| 5500 | 0.5145 | 0.5145 | 0.5145 | 0.4456 | 1.3074 / 1.3075 / 1.3074 | 1.1322 | 1.1332 | 69.79 / -0.0037 |
| 10000 | 0.4594 | 0.4594 | 0.4594 | 0.3979 | 1.15482 / 1.15485 / 1.15483 | 1.00007 | 1.00007 | 1.40 / -0.0011 |

Fixed baselines at every diagnostic step were copy MSE `1.154737`, batch-mean MSE `1.244577`, and
true displacement norm `181.83777`. The best logged genuine rollout still did not beat copy.

### 6.2 Tau-band request

The prompt's prior banded correct-versus-shuffled improvements (`tau=0: 0.4620`, low `0.3250`,
middle `0.0087`, high approximately zero) are consistent with the future-leakage hypothesis, but
those fields are not present in the refreshed run history and no producing artifact/checkpoint is
available. They are retained as **previously reported evidence, not independently reproduced here**.

| Probe | tau=0 | Low | Middle | High | Full rollout |
|---|---|---|---|---|---|
| Correct vs shuffled | prior gap 0.4620 | prior 0.3250 | prior 0.0087 | prior ~0 | measured above |
| Different-video | not logged | not logged | not logged | not logged | not logged |
| Zero | random-tau aggregates only | not banded | not banded | not banded | measured above |
| Learned null | not logged | not logged | not logged | not logged | not logged |
| Condition attention blocked | not implemented/logged | not logged | not logged | not logged | not logged |
| Copy/batch mean | endpoints logged | endpoints logged | endpoints logged | endpoints logged | fixed MSEs logged |

### 6.3 Causal classification

- **Proven by code:** the state contains future target information for every `tau>0`; attention is
  unrestricted; condition use is not structurally mandatory; adaLN residual gates are zero at
  initialization; output ends in affine LayerNorm; no rollout projection or exposure training exists.
- **Strong empirical evidence:** pre-collapse teacher-forced quality greatly exceeded genuine
  rollout quality; correct condition helped before collapse; correct/zero/rolled became
  indistinguishable after collapse; generated displacement vanished.
- **Correlation:** first F_c AGC event at 5350, largest gradient excursion at 5450, first diagnostic
  collapse at 5500.
- **Plausible, not proven:** the excursion damaged gates/attention/output parameters and caused the
  abrupt loss of condition reliance; generated states had already drifted away from the learned
  vector-field support before that event.
- **Ruled out for `t0okr9cb`:** 10% condition dropout (configured 0); moving B/EMA coordinates
  (one frozen bottleneck encoded both endpoints); Gaussian-start mismatch (source was present);
  NaN or skipped-step corruption (none logged).
- **Not established as primary:** final LayerNorm, dataset unpredictability, slot permutation, or
  adaLN-Zero initialization. Each may affect optimization, but none explains the full measured
  temporal pattern alone.

## 7. Off-manifold rollout analysis

The decisive available comparison is functional rather than geometric. At step 5000, the same
checkpoint and fixed batch achieved teacher-forced endpoint MSE `0.10210` but genuine one/eight-step
rollout MSE `1.20616/1.23671`. The teacher-forced prediction was 91.2% better than copy by ratio;
rollout was 4.5%/7.1% worse than copy. More Euler evaluations did not repair it. That establishes a
large exposure/on-path-to-self-generated performance gap before the acute collapse.

The requested matched-tau measurements cannot be reconstructed exactly:

- no `t0okr9cb` checkpoint is local;
- W&B stores scalar aggregates, not `x0,x1,x_tau,hat_x_tau`, source pairing, or sample tensors;
- no exact cached diagnostic features/video batch is local;
- the rollout history does not log intermediate states.

Therefore `||hat_x_tau-x_tau||`, per-token cosine, moment/norm differences, nearest straight-line
distance, teacher-state versus generated-state velocity error, and condition sensitivity on both
state types are **unmeasured**. Calling the failure literally “off-manifold” is a well-supported
hypothesis, not a completed geometric measurement. The exact future target and source pairing are
deterministic in present mode, so an exact diagnostic becomes possible once a verified checkpoint
and fixed batch are available; no approximate synthetic substitute would answer the trained-model
question.

Recommended evaluation-only computation at each Euler time `tau_i`:

1. Cache `x0,x1`, set `x_tf=(1-tau_i)x0+tau_i*x1`, and retain generated `x_hat`.
2. Evaluate identical condition interventions on both states.
3. Report state distance/RMS, token cosine, per-feature mean/std, token norms, velocity
   MSE/cosine/norm ratio, and endpoint estimates.
4. Measure distance to `{(1-t)x0+t*x1 : t in [0,1]}` analytically per sample/token by projecting
   `x_hat-x0` onto `x1-x0` and clipping the scalar to `[0,1]`; also report distance at matched time.

## 8. LayerNorm and architecture findings

`CoarseFlow` concatenates stamped state and condition tokens, processes all `2N_c` tokens through
six bidirectional self-attention/MLP adaLN-Zero blocks, slices the state half, and returns
`LayerNorm(x_state)` (`models.py:482-560`). No mask is supplied to `nn.MultiheadAttention`.

At initialization every block's modulation projection is exactly zero
(`models.py:430-437`), so the residual attention and MLP routes start gated off. This delays all
condition, time, and deep-state use but gates can receive gradients and wake. It is an optimization
property, not evidence that a trained model must ignore condition.

The final LayerNorm removes each hidden token's pre-affine mean and radial scale; shared learned
`gamma,beta` cannot restore arbitrary sample/token-specific magnitude. It is a genuine
representational restriction. But before step 5500 the model produced large, condition-sensitive
displacements and very low teacher-forced error. After collapse both condition sensitivity and
displacement disappeared. Existing moment-correction findings also failed to recover rollout.
Accordingly LayerNorm is **secondary/restrictive, not the primary proven cause**. A matched
`LayerNorm -> Linear` ablation remains useful only after stable source/objective behavior is secured.

Unrestricted attention contributes to optionality: state tokens can use their future-bearing state
stream; condition tokens can read the state even though their outputs are discarded. Directed
cross-attention would make information flow clearer and cheaper to audit, but a mask cannot force
the model to use condition if the state shortcut remains easier.

## 9. Present-to-future feasibility

The path is already implemented, shape-compatible, tested, and empirically attempted.

- `target_present` and `target_future` can come from one EMA bottleneck in the historical joint
  path (`target_bottleneck(detailed)` and `target_abstract`), but the clean implemented experiment
  instead loads one saved online bottleneck and freezes the same module for both endpoints.
- Present and future both have `(B,N_c,D_c)` and share learned slot indices.
- Existing real-data Gate-B artifacts at k=12/k=16 show same-index cosine
  `0.6840/0.6318`, Hungarian cosine `0.6941/0.6493`, identity-Hungarian gaps only
  `0.0102/0.0175`, row-best identity `90.23%/85.50%`, and cross-sample assignment agreement
  `83.47%/76.52%`. This is good evidence that direct same-index subtraction is meaningful, though
  not perfect or a proof of semantic object tracking.
- Online-versus-EMA endpoint RMS in those artifacts is only about `0.008-0.010`, but mixing online
  and EMA endpoints is unnecessary and should remain forbidden.
- The explicit online condition is partly redundant because `x0` is the first state and is also
  passed as condition. It can still provide a stable, unmodified reference after generated state
  moves away from `x0`.
- Starting exactly from observed present removes the Gaussian boundary mismatch and turns the
  target into a direct temporal displacement. It does not make the learned field correct away from
  straight training paths.
- Deterministic MSE flow learns a conditional central tendency when futures are multimodal. A
  stochastic residual may eventually be necessary, but adding it before proving deterministic
  predictable displacement would confound the diagnosis.

For a baseline branch that lacks `df5d959`, the smallest conceptual changes are: add
`TrainConfig.flow_source`; construct both present/future endpoints with one detached target/fixed
bottleneck; select `source=present`; start diagnostic Euler rollout from present; validate the
mode; and record it in provenance. On the current branch these locations already exist at
`config.py:276-284`, `train.py:578-633,721-754,1507-1518,2880-2890`,
`diagnostics.py:395-562`, and the flow-source tests in `tests/test_optimizer_and_flow.py`.

## 10. Ranked improvement options

| Rank | Option | Targeted failure and rationale | Cannot fix / risk | Required isolated experiment and success criterion |
|---:|---|---|---|---|
| 1 | Rollout-aware or self-conditioned training | Directly trains on generated/off-line states; targets the largest pre-collapse TF-rollout gap | May destabilize or reinforce errors; does not force video-specific conditioning | Stable present-source baseline vs a small scheduled generated-state mixture; reduce matched-tau generated-state velocity gap and beat copy/batch mean |
| 2 | Low-tau oversampling | Makes condition/source information load-bearing near the inference boundary | Does not train later accumulated-error states | Change only tau sampler; improve tau=0/low and rollout without degrading endpoint motion |
| 3 | Tau loss reweighting | Similar to oversampling, with explicit control over low-tau gradient mass | Poor weights can hurt high-tau endpoint accuracy | One weighting schedule only; pass low-tau condition gap and full rollout gates |
| 4 | Present-to-future source | Removes Gaussian start mismatch and uses temporal displacement | Already failed once; retains future leakage and exposure bias | Replicate matched noise/present arms with stable-window rule and correct source-specific rollouts |
| 5 | Directed state-to-present cross-attention | Makes condition route explicit/auditable and stops discarded condition tokens reading future state | State self-route can still ignore condition; architecture change | Only after intervention proves ignored condition in a stable checkpoint; correct must beat blocked/shuffled |
| 6 | Block condition tokens from reading state | Removes unnecessary reverse leakage into discarded tokens | Does not stop state stream using its own future-bearing content | Mask-only ablation; preserve or improve rollout and condition gap |
| 7 | `LayerNorm -> Linear` velocity head | Restores sample/token magnitude freedom after normalized hidden state | Does not solve teacher forcing/exposure; new parameters may destabilize | Head-only matched ablation after stable run; improve velocity norm/direction and rollout |
| 8 | Slot alignment / permutation matching | Protects subtraction if slots permute | Existing evidence says mismatch is small; matching may erase intended slot identity | Evaluation-only same-index vs Hungarian displacement first; train only if gap material |
| 9 | Condition dropout 0.10 -> 0 | Removes incentive for unconditional route | Already zero in failed present run; cannot explain it | Relevant only to historical noise baseline; dropout-only arm, require stronger condition gap and rollout |
| 10 | Stochastic residual for present flow | Represents multimodal futures | Adds objective/inference complexity before deterministic competence | Only after deterministic arm beats copy; require diversity plus equal/better proper score |

Ordered ablation sequence: (A) recover/replicate a stable fixed present-source checkpoint with the
expanded evaluation-only probes; (B) low-tau sampling versus baseline; (C) rollout-aware mixture
versus the winning tau recipe; (D) only then test directed attention; (E) test output head
separately; (F) add stochastic residual only after deterministic forecasting passes. Do not bundle
these changes.

## 11. Controlled next experiment

### Arms

1. `flow_source=noise`: fixed Run-60 bottleneck, `x0=epsilon`, `x1=c_future`, inference starts from
   a seeded Gaussian paired to evaluation.
2. `flow_source=present`: same fixed bottleneck, `x0=c_present`, `x1=c_future`, inference starts
   exactly from present.

Keep architecture, final LayerNorm, condition dropout, dataset, seed, batch/order, optimizer,
schedule, duration, F_c initialization hash, evaluation examples, tau/dropout streams, loss, and
all non-source settings identical. Isolate the Gaussian RNG stream so it cannot shift tau,
dropout, data, or model randomness; current `_fixed_flow_randomness` already does this.

### Mandatory corrections to evaluation

- Start the noise arm's rollout from its exact seeded Gaussian, not from `abstract`.
- Save evaluation source noise/pairing and intermediate states.
- Use a source-unique diagnostic batch for “different-video”; keep rolled same-source results
  separately labelled.
- Report tau bands with correct, different-video, zero, learned-null, and attention-blocked
  conditions, plus matched teacher/generated states.
- Pre-register checkpoint diagnostics at least every 500 steps and retain step 2500/5000/7500.
- Declare a run invalid after a persistent predictor collapse or a gradient/AGC event followed by
  loss of condition sensitivity; compare last common stable checkpoints, not corrupted finals.

### Pass/fail gates

At the late stable window (minimum three consecutive diagnostic points), the present arm must:

1. `rollout_endpoint_mse / copy_present_mse <= 0.90` for both 4- and 8-step Euler;
2. `rollout_endpoint_mse / batch_mean_mse <= 0.90`;
3. displacement alignment `>=0.20` and predicted/true displacement norm ratio in `[0.25,1.75]`;
4. correct-condition rollout MSE at least 10% lower than verified different-video and learned-null;
5. generated-state velocity MSE no more than 1.5x teacher-forced velocity MSE in low/mid tau bands;
6. avoid the collapse signature: correct/zero/shuffled equality, alignment near zero, or
   displacement-norm ratio below 0.10;
7. improve the TF-to-rollout degradation ratio materially over the matched noise arm.

These gates are stricter and more mechanism-specific than “training loss decreased.” If neither arm
beats copy, neither wins.

## 12. Exact code locations that would change

No code change is approved or made. For the proposed **evaluation correction/extension only**:

- `diagnostics.py:488-562`: accept/source the correct Gaussian endpoint for the noise arm, return
  intermediate states, add verified different-video/null/blocked-condition interventions, and add
  matched-state metrics.
- `train.py:1793-1853`: retain deterministic evaluation endpoint pairing and pass the selected
  source rather than unconditional `abstract` into rollout.
- `models.py:439-459,525-560`: only if approved, expose an evaluation attention mask; do not alter
  training behavior for the forensic probe.
- `tests/test_optimizer_and_flow.py`: prove source-specific rollout starts, identical non-noise RNG,
  mask semantics, and no checkpoint mutation.

The present-source training implementation itself already lives at `config.py:276-284`,
`train.py:578-633,721-754,1507-1518,2880-2890`, `losses.py:53-80`, and
`diagnostics.py:395-562`.

## 13. Evidence limitations

- Live W&B history was refreshed read-only, but it contains scalar aggregates only.
- The exact `t0okr9cb` step 2500/5000/7500/10000 checkpoints and original persistent log are not
  local. No checkpoint availability or loadability is claimed.
- Exact replay/transplant causality is blocked without checkpoint schema/state, optimizer moments,
  RNG, sampler/data order, original runtime, and fixed batch tensors.
- Tau-band values in the prompt were not present in W&B and could not be independently reproduced.
- W&B rolled conditions were within one source UID, not verified different-video interventions.
- Learned-null and attention-blocked probes require a checkpoint and evaluation wrapper; zero is not
  equivalent to the learned null token because slot/type embeddings are added after substitution.
- Existing Gate-B slot-alignment artifacts use the Run-60 bottleneck on real data, not the trained
  F_c checkpoint. They support endpoint compatibility only.
- No causal conclusion is drawn from the 5350 event alone.

## 14. Final answers to the ten required questions

1. **Is the present condition structurally necessary for the current training objective?** No.
   In noise mode, much of the loss can be solved from future-bearing `z_tau` and tau; in present
   mode, the state already includes present at tau zero and future thereafter. The architecture
   makes condition available, not mandatory.
2. **Can F_c minimize much of its loss without learning present-to-future dynamics?** Yes. Low
   random-tau teacher-forced error alongside failed rollout proves the objective can reward
   on-path reconstruction without a robust generative vector field.
3. **Does `z_tau` create a future-information shortcut?** Yes for every `tau>0`, in both noise and
   present source paths. It is supervised teacher forcing, not inference-available future input.
4. **Is rollout failing because generated states leave the teacher-forced interpolation manifold?**
   Strongly supported as an exposure/on-path generalization failure, but exact geometric departure
   is not yet measured. Treat literal off-manifold distance as a hypothesis pending checkpoint data.
5. **Did the step-5350 event cause conditioning collapse?** Unknown. It is tightly correlated and a
   plausible trigger; causality requires exact replay/transplants. Step 5000 was intact, 5350 was
   first clipping, 5450 the largest excursion, and 5500 first diagnostic collapse.
6. **Is final LayerNorm primary or secondary?** Secondary restriction on current evidence. It is
   real but does not explain the pre-collapse TF/rollout gap or acute loss of conditioning by itself.
7. **Would present-to-future flow address the dominant failure?** It fixes Gaussian-start boundary
   mismatch, but it already failed to fix the dominant measured TF-to-rollout gap in `t0okr9cb`.
   Rollout-aware/low-tau support is now the more direct target.
8. **Would it still suffer future leakage at intermediate tau?** Yes, exactly
   `x_tau=(1-tau)c_present+tau*c_future`.
9. **Are slots sufficiently aligned for direct subtraction?** Probably yes for a controlled first
   experiment: same-index alignment is close to optimal and identity wins most rows at k=12/16.
   It is not perfect, so keep Hungarian diagnostics; do not yet add matching to training.
10. **What is the smallest clean experiment next?** First recover or reproduce a stable fixed
    present/noise source pair with correct source-specific rollout and matched teacher/generated
    diagnostics. If the present arm reproduces the pre-collapse gap, change only low-tau sampling;
    then test a small rollout-aware mixture. Do not combine architecture/head/dropout changes.

| Finding | Evidence | Confidence | Consequence |
| ------- | -------- | ---------- | ----------- |
| Future target enters every nonzero-tau training state | `losses.py:53-80`; `train.py:751-754` | Certain | Teacher-forced loss is not rollout evidence |
| `t0okr9cb` was present-source with zero dropout | Live resolved W&B config | Certain | Noise start and 10% dropout do not explain this run |
| Present-source did not close TF-rollout gap | Step-5000 ratios `0.0884` TF vs `1.0445/1.0710` rollout | High | Target exposure/self-generated states are central |
| Condition was useful before acute collapse | Correct vs rolled/zero endpoint MSE through step 5000 | High | Architecture can use condition; it is not permanently blocked |
| Predictor/conditioning collapsed at 5500 | Correct/zero/rolled equality, alignment near zero, shrinking displacement | High | Post-5500 values diagnose a separate acute failure |
| Gradient/AGC event caused collapse | 5350-5450 temporal adjacency only | Low-to-medium | Requires exact replay/transplant before causal claim |
| Final LayerNorm is restrictive but secondary | Code geometry plus pre-collapse behavior and prior moment tests | Medium-high | Defer head ablation until objective/exposure is isolated |
| Direct slot subtraction is reasonable | Real-data k12/k16 identity vs Hungarian diagnostics | Medium-high | Keep fixed same-index path; monitor alignment |
| Literal off-manifold distance is quantified | Required tensors/checkpoint absent | No evidence | Must remain an explicit open measurement |
