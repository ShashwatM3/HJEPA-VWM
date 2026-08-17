# F_c Architecture and Conditioning Investigation

**Date:** 2026-08-05  
**Scope:** Investigation only. No training, checkpoint download, or implementation change was performed.  
**Evidence labels:** **Confirmed** = current code/tests; **Measured** = local offline measurement;
**Inference** = mathematical or architectural consequence; **Proposal** = unimplemented change.

## 1. Executive conclusion

**Confirmed.** `CoarseFlow` is a six-block, width-256, eight-head adaLN-Zero transformer.
It stamps two 32-token streams (noisy/interpolated future state first, present condition second),
concatenates them into 64 tokens, applies unrestricted bidirectional self-attention, slices the
first 32 tokens, and returns an affine `LayerNorm(256)` result. There is no learned mapping after
that normalization. The output has velocity width only because the transformer width, latent
width, and target velocity width are all `D_c=256`; `f_c_dim` is configured but is not used by
`CoarseFlow`.

**Confirmed.** The condition is available to every returned state token, but is not load-bearing:
there is no mask, cross-attention-only route, auxiliary condition-use objective, or correct-vs-wrong
condition diagnostic. The state stream contains `z_c`, which itself contains the target latent at
nonzero `tau`, so the predictor can reduce flow loss from `z_c` and time while using condition
weakly. Condition dropout is per example at probability 0.10 and substitutes a learned 32-token
null condition. No inference sampler or classifier-free guidance consumer exists.

**Measured.** The default F_c has 7,643,904 trainable parameters. At initialization, all six
adaLN residual gates are exactly zero. Consequently F_c is exactly condition-independent (up to
floating-point effects): gradients into the present condition, attention weights, timestep input,
and null condition were all zero in a synthetic backward probe. Only the final LayerNorm and the
state/slot input route affect the initial output; gate gradients are nonzero and can wake the
blocks. This is an initialization property, not evidence about a trained model.

**Inference.** The LayerNorm-only head imposes a real per-token output manifold. Before affine
parameters, every token is centered and has unit biased variance; with default affine parameters,
its norm is approximately `sqrt(256)=16`. Learned shared `gamma`/`beta` relax this to a globally
warped manifold, but cannot restore arbitrary sample- and token-dependent mean/scale. Direction
can encode some magnitude information indirectly, so the head is not generally constant-norm
after learned affine parameters, but it cannot represent every vector in `R^256`. The restriction
is therefore genuine; whether it is the dominant empirical bottleneck is unproven locally.

**Recommendation.** First add evaluation-only condition interventions and target/output magnitude
metrics. For the smallest objective-aligned architecture change, compare the current model against
`LayerNorm(D_c) -> Linear(D_c, D_c)` with a small Xavier-scaled random initialization (not exact
zero). Keep concatenated attention for this first head ablation. Then test dropout 0.10 versus 0.0.
Only if correct-vs-shuffled condition gaps remain weak should conditioning change to state-query
cross-attention over a separately processed present stream, preserving slot codes. This ordering
isolates output expressivity before paying for a redesign.

## 2. Exact construction and training trace

### 2.1 Source locations

| Concern | Current source |
|---|---|
| Defaults | `config.py:55-88`, especially `n_c`, `d_c`, `f_c_blocks`, `f_c_dim`, `f_c_heads`, `condition_dropout` |
| AdaLN block | `models.py:410-459` |
| Sinusoidal time embedding | `models.py:462-479` |
| `CoarseFlow` construction/forward | `models.py:482-560` |
| Phase-1 module construction | `models.py:896-917` |
| Rectified-flow primitives | `losses.py:31-80` |
| Training call and loss | `train.py:448-463` |
| Optional endpoint/reconstruction route | `train.py:535-562` |
| Backward/AGC route | `train.py:563-575` |
| Dropout-disabled coarse baseline | `diagnostics.py:390-438` |
| Slot/type/optimizer tests | `tests/test_optimizer_and_flow.py:30-115` |
| Checkpoint save/load | `train.py:647-949` |

### 2.2 Constructor and defaults

`CoarseFlow(cfg: ModelConfig)` receives the complete model config. Its effective default geometry is
`N_c=32`, `D_c=256`, 6 blocks, 8 heads, MLP ratio 4, and condition dropout 0.10. `f_c_dim=256`
exists at `config.py:86`, but the implementation consistently uses `cfg.d_c`; changing `f_c_dim`
alone has no effect. `build_phase1_modules()` constructs it as `CoarseFlow(cfg.model)`.

The normal predictive path is:

1. Online bottleneck produces attached `abstract=c_t: (B,32,256)`.
2. EMA target bottleneck produces detached `target_abstract=c_plus: (B,32,256)`.
3. `eps_c ~ N(0,I): (B,32,256)` (or residual-scaled noise in residual mode).
4. `tau_c ~ Uniform[0,1]: (B,)`.
5. `z_c=(1-tau_c)eps_c+tau_c*flow_target: (B,32,256)`.
6. Target velocity `u_c=flow_target-eps_c: (B,32,256)`.
7. `u_c_hat=F_c(z_c,tau_c,c_t): (B,32,256)`.
8. Primary loss is elementwise mean MSE between `u_c_hat` and detached `u_c`.

### 2.3 Complete trainable parameter table

All entries require gradients. Default total: **7,643,904 parameters**.

| Parameter/group | Shape | Count | Initialization/role |
|---|---:|---:|---|
| `null_condition` | `(32,256)` | 8,192 | zeros; substitutes dropped examples |
| `slot_pos` | `(32,256)` | 8,192 | normal std 0.02; shared slot identity |
| `z_type` | `(1,1,256)` | 256 | zeros; state-stream segment code |
| `cond_type` | `(1,1,256)` | 256 | zeros; condition-stream segment code |
| `time_mlp.0` weight/bias | `(1024,256)`, `(1024,)` | 263,168 | PyTorch `Linear` default; then SiLU |
| `time_mlp.2` weight/bias | `(256,1024)`, `(256,)` | 262,400 | PyTorch `Linear` default |
| each block attention QKV weight/bias | `(768,256)`, `(768,)` | 197,376 | PyTorch MHA defaults |
| each block attention output weight/bias | `(256,256)`, `(256,)` | 65,792 | PyTorch MHA defaults |
| each block MLP first weight/bias | `(1024,256)`, `(1024,)` | 263,168 | Linear, GELU |
| each block MLP second weight/bias | `(256,1024)`, `(256,)` | 262,400 | PyTorch `Linear` default |
| each block modulation weight/bias | `(1536,256)`, `(1536,)` | 394,752 | **exact zeros** after SiLU |
| six blocks subtotal | — | 7,100,928 | 1,183,488 per block |
| final `norm.weight`, `norm.bias` | `(256,)`, `(256,)` | 512 | ones/zeros |

The two non-affine block LayerNorms have no parameters. Geometry parameters, norm/bias terms, and
zero-initialized gates are excluded from weight decay/AGC by the shared optimizer predicates; normal
weights use AdamW weight decay 0.05. F_c's peak configured learning rate is `2e-4`.

### 2.4 Time and token construction

The time embedding uses 128 exponentially spaced frequencies for `D_c=256`:

`freq_i = exp(-log(10000) * i / 128)`, then `concat(sin(tau*freq), cos(tau*freq))`.

It is computed in fp32, truncated/padded to 256, cast to the token dtype, and passed through
`Linear(256,1024) -> SiLU -> Linear(1024,256)`. The resulting `(B,256)` vector modulates every
token identically within each block.

Condition dropout executes before token stamping. During training, one Bernoulli decision is drawn
per example. An explicit `(B,)` boolean mask can force behavior in tests/evaluation. Dropped examples
use `null_condition`, but still receive shared `slot_pos` and `cond_type`.

| Sequence range | Token value before block 0 | Meaning | Returned? |
|---|---|---|---|
| `0:32` | `z_c + slot_pos + z_type` | interpolated/noisy future-state slots | Yes |
| `32:64` | `(c_t or null) + slot_pos + cond_type` | present-condition slots | No |

```text
z_c (B,32,256) -------- + slot_pos + z_type -----\
                                                       concat (B,64,256)
c_t/null (B,32,256) --- + slot_pos + cond_type ---/          |
tau (B) -> sin/cos -> time MLP (B,256) ---------------------->| six adaLN-Zero blocks
                                                             | unrestricted self-attention
                                                             v
                                                   slice tokens [0:32]
                                                             |
                                                   affine LayerNorm(256)
                                                             |
                                                   u_hat (B,32,256)
```

### 2.5 Every transformer block

Each of the six identical blocks is pre-norm adaLN-Zero:

1. Non-affine `LayerNorm(256)` on all 64 tokens independently.
2. Time MLP output passes through block-local `SiLU -> Linear(256,1536)` and splits into
   attention shift, scale, gate, then MLP shift, scale, gate.
3. Modulated attention input is `LN(x)*(1+scale)+shift`.
4. Standard eight-head (`head_dim=32`) `nn.MultiheadAttention`, with Q=K=V equal to the complete
   64-token sequence, no attention mask, no causal mask, and no padding mask.
5. Residual: `x <- x + gate_msa * attention`.
6. Non-affine LayerNorm, then the second time-conditioned shift/scale.
7. FFN `Linear(256,1024) -> GELU -> Linear(1024,256)`.
8. Residual: `x <- x + gate_mlp * FFN`.

Thus state tokens attend to state and condition tokens, and condition tokens attend to state and
condition tokens. Corresponding slots have a shared positional code, but there is no bias or mask
that privileges same-slot state-condition attention.

### 2.6 Output and gradient routes

The exact return is `self.norm(x[:, :cfg.n_c])`. The learned affine LayerNorm is the last operation.
No linear, MLP, scale, residual velocity head, or output gate follows it.

Primary flow loss routes gradients into F_c and, when the condition was not dropped and the learned
attention path uses it, into the online bottleneck through `c_t`. It does not route into the EMA
target bottleneck or frozen encoder target branch. Time gradients train the time MLP and every
block modulation layer; `tau` itself is sampled, not a parameter. Null-condition gradients occur
only for dropped examples and only after a condition-to-returned-state path is active. Optional
future reconstruction (`lambda_recon_pred>0`) additionally routes through the endpoint, F_c, and
online bottleneck. Present reconstruction does not train F_c.

**Initialization caveat (measured):** zero adaLN gates make every block identity at step zero.
For a synthetic MSE-on-output backward pass, present-condition, tau-input, null-condition, and first
block attention-weight gradient norms were 0; the first modulation projection received a small
nonzero gradient. The condition becomes usable only as gates wake. This is consistent with
adaLN-Zero, but delays condition learning along with all transformer learning.

## 3. LayerNorm-only output restriction

For one hidden token `h in R^D`, PyTorch LayerNorm computes

`mu = mean(h)`, `sigma² = mean((h-mu)²)`, `q=(h-mu)/sqrt(sigma²+eps)`,
`u_hat = gamma elementwise-multiply q + beta`.

Ignoring epsilon, `mean(q)=0`, `mean(q²)=1`, and `||q||_2=sqrt(D)`. Epsilon makes the variance
slightly below one when hidden variance is small. At default `gamma=1`, `beta=0`, every token is
therefore almost zero-mean with norm 16. The local initialization measurement found maximum absolute
token mean `3.35e-8`, biased standard deviation `0.999993..0.999996`, and norm
`15.99989..15.99994`.

After affine transformation, output mean and norm need not be fixed because the normalized direction
interacts with featurewise `gamma` and `beta`. The transformer can encode some target magnitude by
choosing different directions `q`. Nevertheless `gamma` and `beta` are shared across every batch
element, slot, and time. They cannot apply an arbitrary example-dependent scalar or restore the two
degrees of freedom (per-token mean and radial scale) discarded by normalization. The attainable set
for fixed affine parameters is

`M(gamma,beta) = { beta + gamma elementwise-multiply q : 1^T q=0, ||q||²=D }`,

a transformed `(D-2)`-dimensional manifold in `R^D` (with degeneracies if a gamma entry is zero).
Thus some targets are exactly unrepresentable. A preceding universal transformer does not remove
the final manifold constraint.

For a target token `u`, its irreducible squared error under fixed affine parameters is exactly

`E_min(u;gamma,beta) = min over q: mean(q)=0, mean(q²)=1 of ||u-beta-gamma*q||² / D`.

The dataset-level minimum is the expectation of that quantity, jointly minimized over the one
global `gamma,beta`. With `gamma=g*1` and arbitrary `beta`, a useful closed form is obtained by
decomposing `v=u-beta` into its feature mean and centered part:

`min MSE = mean(v)^2 + (||center(v)||/sqrt(D) - |g|)^2`.

For the default `g=1,beta=0`, varying target token means and centered norms create unavoidable
error. With general featurewise gamma, the constrained optimization is not the simple formula above
and should be solved numerically per token.

The model is therefore not merely inefficient in the strict representational sense. In practice,
learned anisotropic gamma/beta and high dimensionality may make the error small, so empirical
importance must be measured. A cosine-distance plateau near `pi/2` means near-orthogonal directions;
LayerNorm may contribute through optimization or magnitude mismatch, but the restriction alone does
not imply orthogonality. Treat that causal explanation as a hypothesis until a matched head ablation
and target projection-floor measurement support it.

## 4. Available and missing measurements

### 4.1 Local availability

**Measured.** No project `.pt`, `.pth`, or `.ckpt` exists in the repository. The only `.pt` found
is a PyTorch package test asset. Both `/workspace/checkpoints` and `/workspace/ckpt` are absent.
Therefore no trained final LayerNorm gamma/beta, trained prediction distribution, or fixed real-data
target velocity tensor was locally available. No artifact was downloaded.

Consequently the requested real target statistics—token mean/std/norm, norm by tau, video/horizon,
and LayerNorm explainable fraction—are **unresolved**, not zero. The KANBAN record contains historical
aggregate coarse losses/ratios and representation metrics, but not the needed raw velocity tensors
or condition interventions; those aggregates cannot reconstruct this distribution.

### 4.2 Required evaluation-only measurement

On one pinned fixed diagnostic batch and fixed RNG draws, cache `c_t`, detached `flow_target`,
`eps_c`, `tau_c`, `z_c`, target `u_c`, and sample metadata. Report for targets and predictions:

- per-token feature mean, biased feature std, and L2 norm (mean/median/p05/p95);
- the same grouped by tau decile, source video, and horizon;
- target-vs-prediction norm ratio and correlation;
- `E_min` for (a) plain normalized direction, (b) checkpoint gamma/beta, and (c) an optimally fitted
  global gamma/beta on a fit split, evaluated on a held-out split;
- `1 - SSE_projection/SST` as the variance fraction explainable by each LayerNorm manifold.

Do not call the projection fraction “explained” without a held-out split when gamma/beta are fitted.

## 5. Conditioning-dependence audit

### 5.1 What exists

Condition dropout and explicit dropout-mask override exist in `CoarseFlow`. `coarse_baselines()`
forces no dropout and compares model velocity MSE with copy and batch-mean targets. Bottleneck
attention entropy exists, as do shuffled-latent decoder reconstruction and video-gap diagnostics.
Those test bottleneck/decoder information use, not whether F_c uses its present condition.

No current diagnostic performs shuffled, zero, learned-null, wrong-video, attention-blocked, or
state-only F_c evaluation. No F_c attention weights/entropy are returned. There is no current
correct-vs-shuffled condition gap.

### 5.2 Evaluation-only probe matrix

Use identical `z_c`, `tau`, target `u_c`, and model weights for every row. Disable stochastic
dropout via the explicit mask. A temporary evaluation wrapper/hook may apply masks without changing
saved weights; production implementation, if later approved, should expose masks explicitly.

| Probe | Intervention | What it distinguishes |
|---|---|---|
| Correct | true `c_t` | reference |
| Batch shuffled | deterministic derangement of `c_t` | useful sample identity |
| Zero | all-zero pre-stamp condition | generic condition sensitivity |
| Learned null | force all-true drop mask | trained unconditional route |
| Wrong video | source-aware derangement, same `z_c/tau` | video-specific use |
| Blocked state<-condition | attention mask blocks returned state queries from condition keys/values | causal path use |
| State only | remove condition tokens in an evaluation wrapper | reliance on sequence length/content; dimensionally possible for self-attention |
| Condition-only control | replace state content while keeping layout | sanity control; not directly comparable because `z_c` is required by the objective |

For each row report velocity MSE/cosine/norm, one-step endpoint latent MSE/cosine, decoded future
reconstruction where a trained decoder is meaningful, and per-sample `||u_probe-u_correct||` plus
cosine change. Bootstrap confidence intervals over samples/videos.

Interpretation must separate four claims:

- **Uses condition:** output changes under a causal condition intervention.
- **Uses it usefully:** correct condition improves target metrics over shuffled/null.
- **Is perturbed:** output changes, but metrics do not improve (or worsen).
- **Uses video-specific information:** correct beats a different-video condition, ideally after
  controlling horizon/tau, and improves decoded/endpoint future metrics.

## 6. Attention-path alternatives

| Design | Load-bearing potential | Cost/parameters | Slot correspondence and main risk |
|---|---|---|---|
| A. Current bidirectional concat | Low structural guarantee; measurable only by intervention | Attention on 64 tokens; current 7.64M | shared slot code preserved; state can ignore condition, condition updates are discarded |
| B. State tokens attend condition, condition does not attend back | Clear directed condition path, but state self-route can still dominate | similar or lower with a block mask; no required new parameters | slot codes preserved; simplest structural ablation |
| C. Separate condition encoder + state-to-condition cross-attention | Explicit, inspectable condition route | extra condition processing; cross-attn roughly O(Ns*Nc) rather than joint O((Ns+Nc)^2) | correspondence retained via slot codes; risk if condition encoder bottlenecks/overprocesses |
| D. Pooled FiLM/adaLN | Condition enters every block and is easy to ablate | small MLP/modulation increase | pooling destroys slotwise correspondence; easy to ignore through gates |
| E. Cross-attn + FiLM | strongest dual route | highest complexity/parameters | preserves local plus global context; attribution becomes harder |
| F. Input-only vs every-block injection | every-block injection shortens gradient/use paths | repeated modules or shared projections | input-only can be washed out; every-block can overcondition |

The smallest structurally meaningful change after measurement is B: preserve the 64-token layout but
block condition queries from attending back to state (and optionally restrict returned state queries
to a deliberate state-self plus state-condition pattern). However, a mask alone does not force use.
If correct-vs-shuffled gaps are weak, C is the recommended clean architecture: retain state
self-attention, add explicit state-query cross-attention to fixed condition memory in each block,
and expose cross-attention weights or a condition-block mask for diagnostics. Avoid pooled-only FiLM
because the bottleneck's learned slots carry correspondence that pooling discards.

## 7. Candidate velocity heads

| Head | Expressivity and magnitude | Stability/gradients | Extra parameters | Checkpoint compatibility |
|---|---|---|---:|---|
| Direct LN (current) | constrained affine-normalized manifold | stable, but magnitude restricted | 0 | current |
| LN -> Linear | arbitrary affine image; restores token-dependent mean/scale through direction | standard and simple | 65,792 | old checkpoints need explicit migration/init |
| Conditioned LN -> Linear | time/condition-dependent normalization plus free projection | stronger conditioning, more moving parts | projection + modulation | incompatible |
| MLP head | nonlinear full-space map | more capacity and overfit/instability risk | about 0.5M at 4x | incompatible |
| Residual/gated head | controllable deviation from base | gate can delay learning; define base carefully for velocity | varies | incompatible |
| Per-token learned scale | frees slot-global scale only | cheap, still lacks example-dependent scale unless predicted | 32 or 8,192+ | incompatible |

**Proposal.** Use `LayerNorm(256) -> Linear(256,256)` first. It is standard objective-aligned
practice: the normalized hidden state is a representation; a learned projection maps it to the
regression target space. Initialize with a small random Xavier matrix (for example Xavier followed
by a documented scale such as 0.1) and zero bias, then validate initial output/gradient magnitudes.
Full Xavier gives the strongest early transformer gradients but output std near hidden std; a small
scale is more conservative. Exact zero projection produces zero initial velocities but also makes
`dL/dh = W^T dL/du = 0`, blocking early gradients into the transformer, time path, and condition
until the head takes its first update. It is therefore not recommended here. A tiny nonzero random
init avoids that blockade.

## 8. Condition dropout

The likely rationale is classifier-free robustness/unconditional prediction. Current code implements
per-example dropout, which is appropriate for a whole conditioning example; per-token dropout would
change the semantic task and may corrupt slot correspondence. However, no inference sampler, guidance
combination, or other null-condition consumer exists. Eval-mode inference uses the supplied condition
unless an explicit override is passed. Thus unconditional capability is currently trained but unused.

Ten percent dropout may regularize against brittle dependence, but it also explicitly rewards an
unconditional solution when the research question is whether present context is load-bearing. Testing
0% is appropriate. It should not silently replace 10% in historical comparisons: make it a factorial
arm with identical seed/data/RNG policy. If classifier-free guidance is not planned, absence is the
clean default candidate; retain per-example (not per-token) dropout only if unconditional evaluation
shows a concrete benefit.

## 9. Minimal controlled experiment

### 9.1 Required 2x2 factorial

| Arm | Output head | Condition dropout | Purpose |
|---|---|---:|---|
| H0-D10 | current LN-only | 0.10 | exact control |
| H1-D10 | LN + Linear | 0.10 | head main effect |
| H0-D0 | current LN-only | 0.00 | dropout main effect |
| H1-D0 | LN + Linear | 0.00 | interaction / leading candidate |

Use the same flow source, immutable encoder revision, resolved `EncoderSpec`, bottleneck, horizon,
dataset identity/order, seed, optimizer/schedule, width/depth, reconstruction/regularization flags,
physical/global batch, and 15,000 update budget. A historical run is not an adequate control unless
all those fields and code commit match. Prefer at least three seeds after a one-seed screening pass.

### 9.2 Optional second-stage architecture arms

Only if the required matrix shows weak correct-vs-shuffled gaps, compare H1-D0 with (B) directed
masked concatenation and (C) explicit state-to-condition cross-attention. Do not include these in the
first factorial because head and conditioning changes would be confounded and substantially expand
cost.

### 9.3 Pre-registered success criteria

- lower held-out flow MSE with confidence interval and no regression in cosine;
- endpoint latent MSE/cosine improvement across tau deciles, not only near `tau=1`;
- predicted/target norm ratio near 1 with reduced norm calibration error;
- positive correct-vs-shuffled/wrong-video condition gap in velocity, endpoint, and decoded future;
- lower `coarse_vs_copy_ratio` and `coarse_vs_batch_mean_ratio`, targeting existing gates `<=0.70`
  and `<=0.50` respectively;
- finite losses/gradients, no increased skip rate, comparable clipped/raw gradient distributions;
- no deterioration in `c_std`, effective rank, slot rank, cross-video cosine, or reconstruction
  video gap indicating representation collapse.

## 10. File-by-file patch plan (no implementation performed)

1. `config.py`: add an explicit output-head mode and initialization scale; optionally expose a
   conditioning-mode enum. Keep `condition_dropout` explicit. Resolve or remove the unused
   `f_c_dim` contract rather than leaving a false knob.
2. `models.py`: add the selected velocity projection; expose an evaluation-only attention mask or
   clean conditioning mode. Preserve input/output shapes and named gradient boundaries.
3. `diagnostics.py`: add condition intervention metrics and velocity/endpoint magnitude statistics,
   returning flat scalar dictionaries. Reuse the existing fixed batch and already computed tensors.
4. `train.py`: add CLI/config wiring, checkpoint provenance fields, and diagnostic logging. Do not
   add an extra encoder or training forward.
5. `tests/test_optimizer_and_flow.py`: assert head shape/init, nonzero early upstream gradients,
   exact condition-mask semantics, output dimension, optimizer grouping, and checkpoint mismatch.
6. checkpoint tests: require explicit migration for old F_c state dicts; do not silently ignore a
   missing head. Record initialization/migration policy in provenance.
7. `AGENT_FILES/AGENTS.md` and relevant living metric/structure guides: update only alongside an
   approved implementation so docs match code.

## 11. Risks, assumptions, and unresolved questions

- No real fixed diagnostic tensors or checkpoint were local, so target magnitude variability and
  trained condition reliance remain unresolved.
- Historical coarse plateaus are not causal evidence for the output head; representation quality,
  target/noise geometry, optimization, and condition bypass are confounders.
- `z_c` leaks increasing target information as tau rises by design; condition gaps must be reported
  by tau decile or an aggregate can hide reliance at low tau.
- A linear head changes checkpoint schema and initialization dynamics; comparisons require fresh
  matched runs or an explicitly documented migration experiment.
- Cross-attention makes the route visible, not automatically useful. Intervention metrics remain
  mandatory.
- Exact target-manifold projection for featurewise gamma requires a stable constrained solver and
  held-out validation; a naive re-normalization is not the true general-affine optimum.
- Confirm whether future inference will use classifier-free guidance before permanently removing
  null-condition training.

## 12. Final recommended F_c architecture

**Immediate tested candidate:** keep slot/type stamping, time-conditioned six-block transformer,
and current concatenated self-attention; replace the direct return with
`velocity = Linear(LayerNorm(state_hidden))`, small-random Xavier-scaled initialization, zero bias;
set condition dropout to 0 in a matched arm, not as an untested universal change. Add causal
correct/shuffled/null/wrong-video/masked probes first-class to diagnostics.

**Conditional follow-up:** if the head improves magnitude/loss but condition-use gaps remain weak,
use per-block state self-attention plus state-query cross-attention to a present-condition memory.
Keep all 32 condition slots and their shared slot codes; do not pool them away. A directed attention
mask is the cheaper intermediate ablation.

## 13. Go/no-go checklist

- [ ] Real fixed batch and sample/video metadata are available locally.
- [ ] Baseline target and predicted velocity mean/std/norm statistics are recorded by tau.
- [ ] Baseline checkpoint gamma/beta and condition-intervention gaps are recorded.
- [ ] The four required factorial arms differ only in head/dropout.
- [ ] New projection initialization preserves nonzero step-zero gradients into F_c.
- [ ] Checkpoint compatibility/migration fails loudly and is provenance-recorded.
- [ ] Correct condition beats shuffled/wrong-video condition, not merely changes output.
- [ ] Flow/endpoint/reconstruction improve without gradient instability or representation collapse.
- [ ] Only then: **go** on the head/dropout change.
- [ ] Only if condition gaps remain weak: **go** on a directed/cross-attention architecture ablation.

