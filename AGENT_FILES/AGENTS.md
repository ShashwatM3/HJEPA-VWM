# AGENTS.md - HJEPA-VWM living reference for coding agents

This is the entry point for agents working in this repository. It is deliberately
implementation-grounded: use it to build a correct mental model before changing
code, then verify the details by tracing the referenced files yourself.

Do not treat this document as permission to stop reading code. It is a map of the
current system and the intended system, not a substitute for tracing the exact
function, class, and gradient path you are about to edit.

## 0. First rules

1. Read this file first, then read `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md`.
2. Before touching code, read `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md`.
3. Before touching `config.py`, `data.py`, paths, or subsets, read
   `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md`.
4. Before changing architecture, read:
   - `AGENT_FILES/KNOWLEDGE/SUPERVISOR_FEEDBACK_EXPLAINED.md`
   - `AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md`
   - `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md`
   - the relevant phase file under `AGENT_FILES/PHASES/`
5. Do not infer this project from generic ML habits. The important bugs here are
   gradient-routing, target-branch, shape, and diagnostic-contract bugs.
6. Do not put experiment-run history, W&B run narratives, or KANBAN contents in
   this file. This document is about the project architecture and current code.

## 1. What this project builds

HJEPA-VWM is a hierarchical JEPA-flow video world model. It does not train a
pixel diffusion model. Its research claim is that a compact abstract latent can
carry predictive video state, and that a larger detailed latent can later recover
local visual detail under the control of that abstract state.

The intended full v0 hierarchy is:

```text
context clip x_{<=t}
  -> frozen video encoder E
  -> detailed latent e_t
  -> trainable bottleneck B
  -> abstract latent c_t
  -> coarse flow F_c predicts future abstract c_{t+k}
  -> fine flow F_e predicts future detailed e_{t+k}
  -> frame generator renders pixels in frozen VAE latent space
```

The current implemented code is not the full hierarchy. It implements the Phase
1 coarse system plus several gated Phase-1 training mechanisms:

- implemented: data loading, `FrozenEncoder`, `Bottleneck`, `TargetBottleneck`,
  `CoarseFlow`, feature-space reconstruction `Decoder`, losses, diagnostics,
  optimizer/EMA/checkpoint training loop, and tests.
- not implemented: `FineFlow`, Stage 2/3 training, shuffled-c fine-flow tests,
  Phase-3 pixel frame generator, VAE wrapper, inference rollout sampler,
  `eval.py`, and Phase-4 multi-horizon embedding.

When a doc describes `F_e`, frame generation, or multi-horizon prediction, treat
that as intended design until code exists. When code and planning docs differ for
implemented behavior, trace code first and then reconcile the docs explicitly.

## 2. Source precedence

Use this order when sources conflict:

1. Current root implementation files for implemented behavior:
   `config.py`, `data.py`, `models.py`, `losses.py`, `diagnostics.py`,
   `train.py`, `make_subset.py`, and tests.
2. `AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md` for current architecture intent.
3. `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` for shapes, terms, and planned
   stage semantics.
4. Active `AGENT_FILES/PHASES/PHASE_N.md` for work sequencing when the human
   explicitly asks to execute a phase.
5. `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` for code layout and style.
6. `README.md` and human-facing `YOUR_FILES/` notes for orientation only.

Important drift to know:

- The stale planning docs often mention 30k Phase-1 steps and older operating
  defaults. Current code defaults are in `config.py`: Phase 1 has
  `stage1_steps = 15000`, `warmup_steps = 1500`, `grad_clip = 0.5`, and
  `grad_skip_threshold = 150.0`.
- `BRIEF_V0_3.md` includes empirical run discussion. Do not copy run history
  into this entry file.
- `PHASE_2.md` still has an open detailed-target geometry decision. Do not
  implement `F_e` until that ambiguity is resolved by the human.

## 3. Repository map

Root implementation files:

| Path | Role |
|---|---|
| `config.py` | Dataclass configuration, path contract, dimensions, optimizer/loss knobs. |
| `data.py` | SSv2 video dataset and dataloader. Produces context/target clip pairs. |
| `make_subset.py` | Builds `ssv2_tiny` as symlinks plus `manifest.json`. |
| `models.py` | All `nn.Module` classes currently implemented. |
| `losses.py` | Pure tensor losses and detach helper. No parameters. |
| `diagnostics.py` | Collapse metrics, baseline comparisons, AGC, weight-decay grouping. |
| `train.py` | Stage-0 sanity, Stage-1 training, CLI, optimizer, EMA, checkpoints. |
| `parse_logs.py` | Parses `step=N {dict}` console logs into JSON. |
| `run_history.py` | W&B Public API export/report helper for logged metrics. |
| `requirements.txt` | Runtime and dev dependencies. |
| `pyproject.toml` | Black and Ruff configuration. |
| `tests/` | Unit tests for contracts, losses, optimizer grouping, AGC, decoder, modes. |

Agent and architecture docs:

| Path | Role |
|---|---|
| `AGENT_FILES/AGENT-BEHAVIOUR/PROTOCOL.md` | How agents operate, when to ask, phase discipline. |
| `AGENT_FILES/AGENT-BEHAVIOUR/CODE_DESIGN.md` | Flat-file layout, naming, docstrings, detach rules. |
| `AGENT_FILES/SETUPS/VOLUME_LAYOUT.md` | RunPod `/workspace` data/checkpoint/cache layout. |
| `AGENT_FILES/KNOWLEDGE/BRIEF_V0_3.md` | Current architecture brief. |
| `AGENT_FILES/KNOWLEDGE/UNDERSTANDING.md` | Expanded concept, shape, loss, stage, stop-grad reference. |
| `AGENT_FILES/PHASES/PHASE_1.md` | Original Phase-1 build spec, partly superseded by code. |
| `AGENT_FILES/PHASES/PHASE_2.md` | Planned fine-flow work. Has an unresolved geometry decision. |
| `AGENT_FILES/PHASES/PHASE_3.md` | Planned frame-generator and eval work. |
| `AGENT_FILES/PHASES/PHASE_4.md` | Deferred multi-horizon extension. |

`KANBAN/` and `AGENT_FILES/KANBAN/` exist, but their contents are not part of
this living reference. Use them only if the human specifically asks for
task-history or current task-management context.

## 4. Canonical terminology and tensor shapes

Names in code follow the symbol map from `CODE_DESIGN.md`.

| Symbol | Code name | Meaning | Shape |
|---|---|---|---|
| `x` | `context_clip` | Context video window ending at time `t` | `(B, 8, 3, 256, 256)` |
| `x_{<=t+k}` | `target_clip` | Future clip window ending `horizon_k` frames later | `(B, 8, 3, 256, 256)` |
| `E` | `encoder` | Frozen V-JEPA 2 ViT-L/16 video encoder | module |
| `e_t` | `detailed` | Detailed context latent from frozen encoder | `(B, 1024, 1024)` |
| `B` | `bottleneck` | Trainable compressor from detailed to abstract | module |
| `c_t` | `abstract` | Current abstract latent | `(B, 32, 256)` by default |
| `B_EMA` | `target_bottleneck` | EMA copy of `B`, never backpropagated | module |
| `e_plus` | `target_detailed` | Future detailed latent from frozen encoder | `(B, 1024, 1024)` |
| `c_plus` | `target_abstract` | Future abstract EMA target | `(B, 32, 256)` |
| `F_c` | `coarse_flow` | Rectified-flow velocity predictor for `c_plus` | module |
| `D` in current code | `decoder` | Feature reconstruction decoder `c -> e_hat`; not the Phase-3 pixel generator | module |
| `c_hat` | `c_hat` / endpoint | One-step estimate of future abstract latent | `(B, 32, 256)` |

Shape derivation:

- Encoder patch size is 16, tubelet size is 2.
- Each 8-frame clip becomes 4 temporal tubelets.
- Each 256x256 frame has a 16x16 spatial patch grid.
- `N_ctx = (8 / 2) * (256 / 16)^2 = 4 * 256 = 1024`.
- `D_e = 1024` comes from `facebook/vjepa2-vitl-fpc64-256`.
- `N_c = 32` and `D_c = 256` are the abstract bottleneck defaults.

Do not introduce functions that touch tensors without shape-contract docstrings.
When modifying tensor paths, trace shape assertions in `models.py`,
`losses.py`, and tests.

## 5. Data path, from disk to batch

Runtime assets live outside the repo under `/workspace` on RunPod:

```text
/workspace/hierarchal-jepa-flow-world-model/  # git repo, code only
/workspace/data/ssv2/                         # full SSv2 symlink layout
/workspace/data/ssv2_tiny/                    # symlink smoke subset
/workspace/ssv2_raw/                          # raw .webm backing files
/workspace/checkpoints/                       # training checkpoints
/workspace/hf_cache/                          # Hugging Face / diffusers cache
```

Only `JEPA_DATA_ROOT` overrides the dataset parent path. Do not add path
auto-detection.

`make_subset.py`:

1. Loads `<data_root>/ssv2/labels.json` as `video_id -> class label`.
2. Groups existing `.webm` symlinks by class for `train` and `validation`.
3. Selects deterministic per-class subsets.
4. Creates idempotent symlinks in `<data_root>/ssv2_tiny`.
5. Writes `manifest.json`.

`data.py`:

1. `SSV2Dataset` indexes `.webm` files in the selected split.
2. `_open_video_reader` opens decord `VideoReader(..., num_threads=1)`. The
   single-thread setting avoids known VP9/decord packet errors while dataloader
   workers still parallelize across videos.
3. `_window_indices` selects two 8-frame stride-2 windows:
   - context indices: `start + i * frame_stride`
   - target window ends at `start + (T - 1) * stride + horizon_k`
   - too-short videos are padded by clamping to the last frame.
4. `_decode_frames` decodes only the 16 needed frame indices, not the whole
   video.
5. Frames become float tensors in `[0, 1]`, channel-first.
6. `_resize_shorter_side` resizes so the shorter side is 256.
7. `_crop` uses random crop for train and center crop for validation.
8. `_color_jitter` applies one brightness/contrast/saturation sample to the
   combined context+target stack for train only.
9. `_normalize_encoder` applies V-JEPA/ImageNet mean/std:
   `(0.485, 0.456, 0.406)` and `(0.229, 0.224, 0.225)`.
10. The item returns `(context_clip, target_clip)`, both
    `(8, 3, 256, 256)`.

No horizontal flip, temporal flip, rotation, tubelet dropout, or `[-1, 1]`
normalization belongs on the encoder path. VAE `[-1, 1]` normalization is a
future Stage-4 concern, not current code.

## 6. Current model components

All current trainable modules live in `models.py`.

### FrozenEncoder

`FrozenEncoder` wraps `transformers.AutoModel.from_pretrained(
"facebook/vjepa2-vitl-fpc64-256", attn_implementation="sdpa")`.

Contracts:

- Calls `model.get_vision_features(clip)`.
- Input is `(B, 8, 3, 256, 256)`, already encoder-normalized.
- Output is `(B, 1024, 1024)`.
- All parameters have `requires_grad=False`.
- `.train()` is overridden to keep the wrapped model in eval mode.
- `forward` is decorated with `torch.no_grad()`.
- The same instance encodes context and target clips.

Never put encoder parameters in an optimizer. Never add a target encoder.

### Bottleneck

`Bottleneck` maps `e_t` or `e_plus` from `(B, 1024, 1024)` to
`(B, N_c, D_c)`.

Pipeline:

1. `in_proj`: linear `D_e=1024 -> bottleneck_mixer_dim=256`.
2. Reshape temporal-major tokens into `(B * 4, 256, 16, 16)`.
3. Apply two shared `ConvNeXtBlock`s per temporal slot.
4. Flatten back to `(B, 1024, 256)`.
5. `to_kv`: linear `256 -> D_c=256`.
6. Cross-attention from `N_c=32` learned query slots to all 1024 memory tokens.
7. Residual output MLP and final LayerNorm.

Important current initialization:

- Query slots are initialized with `nn.init.orthogonal_`.
- The last layer of `out_mlp` is zero-initialized and tagged
  `is_zero_init = True`.
- Zero-init and learned geometry parameters are excluded from AGC and weight
  decay by predicates in `diagnostics.py`.

`Bottleneck(..., return_attn=True)` returns per-head attention weights for
diagnostics: `(B, heads, N_c, N_ctx)`.

### TargetBottleneck

`TargetBottleneck` is a deepcopy of `Bottleneck`:

- It is initialized from the online bottleneck.
- All parameters have `requires_grad=False`.
- It stays in eval mode.
- It updates only through `_update_ema` in `train.py`.
- Its `forward` runs under `torch.no_grad()` and returns `as_target(abstract)`.

It produces `c_plus = B_EMA(e_plus)`, the stop-gradient target for the coarse
flow. It must never receive gradients or optimizer steps.

### CoarseFlow

`CoarseFlow` predicts rectified-flow velocity in abstract space:

```text
F_c(z_c, tau_c, c_t) -> u_c_hat
```

Inputs:

- `z_c`: noised target latent `(B, N_c, D_c)`.
- `tau_c`: flow time `(B,)`.
- `abstract`: current `c_t` condition `(B, N_c, D_c)`.

Structure:

- learned `null_condition` for condition dropout.
- learned `slot_pos`, `z_type`, and `cond_type` embeddings that stamp slot and
  stream identity before concatenation.
- sinusoidal timestep embedding plus MLP.
- six `AdaLNBlock`s by default.
- each `AdaLNBlock` uses LayerNorm without affine, multi-head self-attention,
  MLP, and zero-initialized modulation producing six shift/scale/gate tensors.
- concatenates `[z_c stream, condition stream]` into `2 * N_c` tokens.
- returns only the first `N_c` output tokens, normalized.

Condition dropout:

- During training, each example drops the condition with probability
  `condition_dropout = 0.10`.
- Diagnostics can pass `condition_drop=_no_drop(...)` to force real conditioning.

There is no horizon embedding in current code. Phase 4 will need a deliberate
extension point here.

### Decoder

`Decoder` is the current feature reconstruction decoder. It is not the planned
Phase-3 pixel/frame generator.

Purpose:

- Decode an abstract latent `(B, N_c, D_c)` back to frozen detailed features
  `(B, N_ctx, D_e)`.
- Provide optional reconstruction pressure so `c_t` remains information-rich.
- Support both present reconstruction `D(c_t) -> e_t` and prediction-side
  reconstruction `D(c_hat) -> e_plus`.

Structure:

- `kv_proj`: projects abstract slots to decoder width.
- deterministic fixed 3D tubelet position codes are registered as the
  non-trainable buffer `fixed_pos`.
- initial cross-attention queries are normalized fixed positions, values come
  from `c`.
- `DecoderBlock`s use fixed positions as query information but do not add
  position as output content.
- output projects decoder width back to `D_e`.

Invariant: the decoder may know where output tubelets are, but it must not use
a learned per-output-token content template. Tests enforce that `fixed_pos` is
a buffer and that zero latent cannot emit position-specific content.

## 7. Flow math and implemented loss functions

All current flow losses use rectified flow:

```text
eps ~ N(0, I)
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * eps + tau * target
u_target = target - eps
L_flow = mean((u_hat - u_target)^2)
```

`losses.py` owns the pure tensor primitives:

- `as_target(x)`: centralized stop-gradient helper, returns `x.detach()`.
- `interpolate(target, eps, tau)`: computes `(1 - tau) * eps + tau * target`.
- `velocity_target(target, eps)`: computes `target - eps`.
- `flow_matching_loss(u_hat, u_target)`: mean squared velocity error.
- `residual_target(target_future, target_present)`: returns detached
  `Delta = target_future - target_present` and scalar `sigma = std(Delta)`.
- `reconstruction_loss(pred_detailed, target_detailed, mode)`.
- `variance_floor(abstract)`.
- `sigreg_loss(abstract)`.
- `covariance_floor(abstract)`.
- `slot_diversity_loss(abstract)`.

### Coarse full-latent objective

Default mode predicts the full future abstract target:

```text
flow_target = c_plus
eps_c = randn_like(c_plus)
z_c = interpolate(c_plus, eps_c, tau_c)
u_c = c_plus - eps_c
u_c_hat = F_c(z_c, tau_c, c_t)
L_flow = mean((u_c_hat - u_c)^2)
```

Gradient from `L_flow` reaches `B` through `c_t` and reaches `F_c`. It does not
reach the frozen encoder, `B_EMA`, `e_plus`, or `c_plus`.

### Residual prediction mode

When `cfg.train.predict_residual` is true:

```text
c_present_ema = B_EMA(e_t)
Delta = c_plus - c_present_ema
sigma = std(Delta)
eps_c = sigma * randn_like(Delta)
flow_target = Delta
```

`F_c` predicts residual velocity. The prediction-side reconstruction endpoint is
converted back to a future latent by `c_hat = c_t + Delta_hat`.

The copy/no-change baseline remains comparable because "copy" becomes
"predict zero residual".

### Reconstruction loss

`reconstruction_loss` always detaches the target detailed features.

Modes:

- `cosine` (default): L2-normalize each tubelet vector along `D_e`, then
  return `mean(1 - cosine(pred, target))`.
- `relative_mse`: return raw MSE divided by `Var(target)`, retained as an
  explicit legacy comparison mode.

Current reconstruction paths:

- Present anchor: `decoder(c_t)` vs `e_t`. Trains `D` and `B`, not `F_c`.
- Prediction-side anchor: `decoder(c_hat)` vs `e_plus`. Trains `D`, `F_c`,
  and `B` through the coarse conditioning path.
- Diagnostic readouts also score `decoder(c_plus)` and `decoder(c_hat)` under
  `no_grad`.

### Variance floor

`variance_floor(abstract)`:

1. Reshapes `c_t` to `(B, N_c * D_c)`.
2. Computes per-coordinate std across batch.
3. Applies hinge `max(0, std_target - std)`.
4. Averages over coordinates.

Formula:

```text
L_var = mean_j max(0, 1.0 - Std(c_j))
```

It prevents constant-code collapse. It does not decorrelate dimensions or force
distinct slots.

### SIGReg, covariance, and slot losses

These exist in code but are off by default:

- `sigreg_loss`: stochastic projection/BHEP-style normality statistic pushing
  pooled `c_t` rows toward isotropic `N(0, I)`. It uses a per-step generator in
  `train_step` so logging it does not perturb the global RNG when its weight is
  zero.
- `covariance_floor`: VICReg-C off-diagonal covariance penalty on pooled
  `B * N_c` rows over `D_c` dimensions.
- `slot_diversity_loss`: centers slots within each video, normalizes residual
  slot vectors, and penalizes squared off-diagonal slot cosine similarities.

Do not assume these should be turned on. Their weights default to zero. If you
change them, explain the gradient effect and watch the core prediction metrics.

## 8. Exact current training step

`train.train_step` is the authoritative implementation.

Inputs:

- `batch = (context_clip, target_clip)`.
- `modules = (encoder, bottleneck, target_bottleneck, coarse_flow, decoder)`.
- optimizer over `B`, `F_c`, and `Decoder`.

Normal prediction mode:

1. Move clips to device. In `present_recon_only`, skip moving/using
   `target_clip`.
2. `optimizer.zero_grad(set_to_none=True)`.
3. Enter CUDA bf16 autocast when available and configured.
4. `_coarse_forward`:
   - with no grad: `detailed = encoder(context_clip)`.
   - trainable: `abstract = bottleneck(detailed)`.
   - with no grad: `target_detailed = encoder(target_clip)`.
   - target branch: `target_abstract = target_bottleneck(target_detailed)`.
5. Build flow target:
   - full-latent: `flow_target = target_abstract`, `eps_c = randn_like`.
   - residual: target-present from `B_EMA(detailed)`, `Delta`, scaled noise.
6. Sample `tau_c = rand(B)`.
7. Build `z_c`, `u_c`, and `u_c_hat`.
8. Compute `flow_loss`.
9. Always compute for logging: `var_loss`, `cov_loss`, `slot_loss`,
   `sigreg_l`.
10. Start total loss as `flow_loss`.
11. Add `lambda_var * var_loss`.
12. Add optional active terms:
    - `lambda_cov * L_cov` if `lambda_cov > 0`.
    - `lambda_slot * L_slot` if `lambda_slot > 0`.
    - `lambda_sigreg * sigreg_scale * L_sigreg` if `lambda_sigreg > 0`.
13. If reconstruction anchors are active, compute a linear `recon_scale`.
14. If `lambda_recon > 0`, add present reconstruction.
15. If `lambda_recon_pred > 0`, build one-step endpoint:
    `endpoint = z_c + (1 - tau_c) * u_c_hat`; in residual mode,
    `c_hat = abstract + endpoint`, otherwise `c_hat = endpoint`; add
    prediction-side reconstruction.
16. Exit autocast and call `loss.backward()`.
17. Apply module-specific AGC to `B`, `F_c`, and `Decoder` if enabled.
18. Apply global `clip_grad_norm_` with `grad_clip`.
19. If the returned norm is non-finite or greater than
    `grad_skip_threshold`, zero grads and skip `optimizer.step()`.
20. Otherwise step optimizer.
21. Compute EMA momentum with `ema_cosine`.
22. If the optimizer step was not skipped, update `B_EMA` from `B`.
23. Return a flat metrics dict.

Present-only reconstruction mode:

- `_present_forward` encodes only the context.
- `flow_loss` is zero.
- Future branch, residual mode, `F_c`, and `lambda_recon_pred` are disabled.
- `lambda_recon` must be positive or `finalize_training_config` raises.
- Non-prediction regularizers still follow their configured weights.

## 9. Total objective in code

In normal mode:

```text
L_total =
    L_flow
  + lambda_var * L_var
  + [lambda_cov * L_cov if lambda_cov > 0]
  + [lambda_slot * L_slot if lambda_slot > 0]
  + [lambda_sigreg * sigreg_scale * L_sigreg if lambda_sigreg > 0]
  + [lambda_recon * recon_scale * L_recon if lambda_recon > 0]
  + [lambda_recon_pred * recon_scale * L_recon_pred if lambda_recon_pred > 0]
```

`sigreg_scale` and `recon_scale` are linear ramps from 0 to 1 over their warmup
step counts.

Terms may be computed for logging even when their weights are zero. Do not
mistake a logged scalar for an active gradient source.

## 10. Optimizer, clipping, EMA, and checkpoints

Optimizer:

- `make_optimizer` creates AdamW over `B`, `F_c`, and `Decoder`.
- The decoder is always included so checkpoints are consistent; when no
  reconstruction loss is active, it receives no gradients and does not move.
- Each module is split into decay and no-decay groups by
  `diagnostics.partition_decay_params`.
- Decayed: genuine Linear/Conv weight matrices.
- No-decay and no-AGC: 1-D params, biases, LayerNorm weights, learned geometry
  params named `queries`, `null_condition`, `slot_pos`, `z_type`, `cond_type`,
  and submodules tagged `is_zero_init`.

LR schedule:

- `lr_scale`: linear warmup for `warmup_steps`, then cosine decay to zero over
  `stage1_steps`.
- `apply_lr_schedule` updates every optimizer group from peak base LRs.
- `peak_base_lrs` rebuilds base LRs from the current config so resuming does not
  double-apply saved scheduled LRs.

AGC and global clipping:

- `adaptive_gradient_clip` enforces per-tensor `||g|| <= lambda * (||w|| + eps)`
  on eligible tensors.
- Current default AGC lambdas: `B=0.20`, `F_c=0.10`, `Decoder=0.20`.
- Global `clip_grad_norm_` then clips all trainable params to `grad_clip=0.5`.
- `grad_norm` in `train_step` is the norm returned by `clip_grad_norm_` before
  the global rescale, after AGC.
- `gradient_health` reports a post-clip norm and should not be read as the raw
  gradient magnitude.

EMA:

```text
m(step) = end - (end - start) * 0.5 * (1 + cos(pi * clamp(step / total, 0, 1)))
B_EMA <- m * B_EMA + (1 - m) * B
```

Defaults: start `0.996`, end `0.9999`, denominator `105000`.

EMA updates only after successful optimizer steps. If a step is skipped, the EMA
target does not move.

Checkpoints:

- `save_checkpoint` writes `global_step`, `bottleneck`, `target_bottleneck`,
  `coarse_flow`, `decoder`, `optimizer`, and serialized config.
- The frozen encoder is never checkpointed; reload it from Hugging Face.
- `load_checkpoint` supports older checkpoints without decoder state.
- It rejects checkpoints from the old learned-query decoder architecture.
- It skips incompatible optimizer state group counts with a warning and keeps
  the model weights loaded.

## 11. Diagnostics and what each one proves

Diagnostics are pure functions in `diagnostics.py` and are run on a fixed
validation batch in `train.run_diagnostics`.

Representation health:

- `variance_stats(c_t)` logs `c_std_mean`, `c_std_median`,
  `c_dead_dim_frac`.
- `cross_video_cosine(c_t)` logs mean pairwise cosine between batch examples.
  High values indicate video-independent collapse.
- `effective_rank(c_t)` pools batch and slots, computes covariance over
  `D_c`, then logs `exp(entropy(normalized_eigenvalues))`.
- `slot_diversity_rank(c_t)` computes within-video effective rank over the
  `N_c` slots.
- `attention_entropy(bottleneck, detailed)` computes normalized per-head
  cross-attention entropy. It is useful but weaker than actual slot/rank
  metrics.

Prediction baselines:

- `coarse_baselines` disables condition dropout and compares `F_c` against:
  - copy/no-change baseline.
  - batch-mean future target baseline.
- It logs model/copy/batch-mean losses and ratios.
- In residual mode, copy means predict zero residual.

Reconstruction readouts:

- `L_recon_present`: `decoder(c_t)` vs `e_t`.
- `L_recon_cplus`: `decoder(c_plus)` vs `e_plus`.
- `L_recon_chat`: `decoder(c_hat)` vs `e_plus`.

Gradient and stability:

- `grad_norm`: per-step post-AGC/pre-global-rescale norm from training.
- `grad_skipped`: whether optimizer/EMA update was skipped.
- `instability_warn`: `grad_norm` and `L_flow` are both above warning
  thresholds.
- `agc_*`: module-level AGC ratios and clipped tensor counts.
- `grad_global_norm_postclip`, `grad_has_nan`, `grad_param_count`: diagnostic
  pass gradient health.

Future Phase-2/3 diagnostics, not implemented yet:

- shuffled-c test.
- zero-c condition test.
- teacher-vs-predicted fine-flow gap.
- decoder dependency test with shuffled `e_hat`.
- full `eval.py`.

## 12. Configuration defaults and CLI overrides

`config.py` is the source of current implemented defaults.

Model defaults:

| Field | Default |
|---|---|
| `encoder_repo` | `facebook/vjepa2-vitl-fpc64-256` |
| `encoder_frozen` | `True` |
| `encoder_patch` / `encoder_tubelet` | `16` / `2` |
| `t_ctx`, `h`, `w` | `8`, `256`, `256` |
| `n_ctx`, `n_tgt` | properties, both `1024` |
| `n_c`, `d_c`, `d_e` | `32`, `256`, `1024` |
| bottleneck blocks / heads | `2` ConvNeXt blocks, `8` cross-attn heads |
| `f_c_blocks`, `f_c_dim`, `f_c_heads` | `6`, `256`, `8` |
| `condition_dropout` | `0.10` |
| reconstruction decoder dim / blocks / heads | `256`, `2`, `8` |

Training defaults:

| Field | Default |
|---|---|
| `global_batch` | `64` |
| `stage1_steps`, `max_steps` | `15000`, `15000` |
| `lr_bottleneck`, `lr_coarse_flow`, `lr_decoder` | `1e-4`, `2e-4`, `1e-4` |
| `warmup_steps`, `total_latent_steps` | `1500`, `105000` |
| `adam_betas`, `weight_decay` | `(0.9, 0.95)`, `0.05` |
| `grad_clip`, `grad_skip_threshold` | `0.5`, `150.0` |
| `agc_enabled` | `True` |
| `agc_lambda_bottleneck`, `agc_lambda_coarse_flow`, `agc_lambda_decoder` | `0.20`, `0.10`, `0.20` |
| `ema_m_start`, `ema_m_end`, `ema_schedule_steps` | `0.996`, `0.9999`, `105000` |
| `lambda_var`, `var_floor_std_target` | `0.10`, `1.0` |
| `lambda_sigreg`, `lambda_cov`, `lambda_slot` | `0.0`, `0.0`, `0.0` |
| `lambda_recon`, `lambda_recon_pred` | `0.0`, `0.0` |
| `recon_loss_mode`, `recon_warmup_steps` | `cosine`, `2000` |
| `present_recon_only`, `predict_residual` | `False`, `False` |
| `horizon_k`, `frame_stride` | `4`, `2` |
| `precision` | `bf16` |
| `log_every`, `diag_every`, `checkpoint_every` | `50`, `500`, `2500` |

Data/path defaults:

| Field | Default |
|---|---|
| `data_root` | `os.environ["JEPA_DATA_ROOT"]` or `/workspace/data` |
| `dataset` | `ssv2_tiny` |
| `num_workers`, `pin_memory` | `8`, `True` |
| `checkpoint_dir` | `/workspace/checkpoints` |
| `hf_cache_dir` | `/workspace/hf_cache` |
| `seed` | `42` |

Current `train.py` CLI flags include data selection, steps, resume, seed,
logging cadence, `horizon_k`, all optional loss weights, reconstruction modes,
residual/present-only switches, LRs, AGC toggles, gradient skip threshold,
decoder capacity, `n_c`, and checkpoint directory. Read `parse_args()` before
adding or changing any experiment knob.

## 13. Stage and phase status

Be precise about "stage" vs "phase":

- Training stages are the model curriculum.
- Agent phases are implementation work packages.

Current code:

| Training stage | Implemented? | Notes |
|---|---|---|
| Stage 0 sanity | yes | `python train.py --stage0-only` loads encoder and runs one synthetic step. |
| Stage 1 coarse dynamics | yes | `B`, `B_EMA`, `F_c`, optional reconstruction/regularizer knobs. |
| Stage 2 fine teacher forcing | no | Requires `FineFlow`; Phase 2 spec only. |
| Stage 3 predicted-coarse fine | no | Requires detached `c_hat` into `F_e`; Phase 2 spec only. |
| Stage 4 pixel/frame generation | no | Requires VAE, frame generator, inference/eval; Phase 3 spec only. |
| Phase 4 multi-horizon | no | Deferred; no horizon embedding in code. |

Do not start a later phase unless the human explicitly asks. Do not quietly
implement planned modules while fixing current Phase-1 code.

## 14. Planned full pipeline, not current code

This is the intended end state from the phase docs.

Phase 2 adds `FineFlow`:

- predicts velocity toward `e_plus`.
- conditions on `e_t` plus `c_cond`.
- Stage 2 uses `c_cond = c_plus`.
- Stage 3 uses `c_cond = as_target(c_hat)` after a ramp.
- `L_e` must not backpropagate into `F_c` through `c_hat`.
- shuffled-c and zero-c tests prove that detailed prediction actually depends
  on abstract state.

Phase 3 adds frame generation:

- frozen `diffusers.AutoencoderKL` VAE.
- future frame target in VAE latent space.
- frame generator trains with rectified-flow loss in patched VAE latent space.
- latent stack and VAE are frozen; only frame generator trains.
- decoder dependency tests prove frames depend on predicted detailed state.

Phase 4 adds multi-horizon coarse prediction:

- horizon set `(4, 8, 16, 32)`.
- per-sample horizon sampling.
- learned horizon embedding `h_k`.
- one shared `F_c` handles all horizons.
- data pipeline must sample target windows ending at each selected horizon.

Again: none of these are currently implemented. Check for classes/functions
before assuming they exist.

## 15. Non-negotiable invariants

Do not violate these without explicit human approval:

1. The encoder is frozen, shared, and never optimized.
2. There is no target encoder. Only `B` has an EMA copy.
3. `B_EMA` never receives gradients and never enters an optimizer group.
4. All target branch outputs are stop-gradient.
5. Use `as_target()` for detach boundaries; do not scatter unexplained
   `.detach()` calls.
6. `c_t` on the conditioning path is not detached during latent training.
7. In future Stage 3, `c_hat` must be detached before feeding `F_e`.
8. In future Stage 4, `e_hat` and the latent stack must be detached/frozen
   before training the pixel generator.
9. Encoder input normalization is ImageNet/V-JEPA stats, not `[-1, 1]`.
10. No tubelet dropout on frozen encoder inputs.
11. SSv2 direction-sensitive transforms must not include horizontal/temporal
    flips unless a human approves a changed data contract.
12. `c_t` must remain a bottleneck. Do not casually widen `N_c`, `D_c`, or add
    bypasses that let future prediction ignore the abstract path.
13. Diagnostics are part of the architecture. A component is not done if the
    bypass/collapse test that proves it is real is absent.
14. Changing locked dimensions, gradient routing, or phase sequencing requires
    escalation.
15. Do not import KANBAN or run-history content into this architectural entry
    file.

## 16. How to verify your understanding before edits

For any code change, trace the relevant path in this order:

1. Find the public function/class with `rg -n "class|def"`.
2. Read its docstring and call sites.
3. Trace tensor shapes from `config.py`.
4. Identify which loss terms can reach it.
5. Identify detach/EMA/frozen boundaries.
6. Check diagnostics or tests that assert the intended contract.
7. Add or update tests when you change a contract.

Useful local checks:

```bash
python -m py_compile config.py data.py models.py losses.py diagnostics.py train.py make_subset.py
pytest -q
python -c "from models import smoke_test_models; smoke_test_models()"
python -c "from diagnostics import smoke_test_diagnostics; smoke_test_diagnostics()"
```

RunPod checks that may download or require data:

```bash
python -c "from data import smoke_test_dataloader; smoke_test_dataloader()"
python -c "from models import smoke_test_encoder; smoke_test_encoder()"
python train.py --stage0-only
```

Use `pytest tests/test_phase1_contract.py -q` when touching config, subset, or
flat-file deliverables. Use targeted tests in `tests/test_*` when touching AGC,
optimizer grouping, reconstruction, decoder, SIGReg, or present-only mode.

## 17. File-level implementation guide

`config.py`:

- Holds dataclasses and path defaults.
- Includes properties for token geometry.
- Avoid inline magic constants in hot paths. Add config fields first.
- CLI overrides are applied in `train.main`, not here.

`data.py`:

- Owns only SSv2 loading and preprocessing.
- Keep context/target transforms shared where intended.
- Do not change video sampling semantics without updating docs and tests.

`make_subset.py`:

- Symlink-only. Never copy or re-encode video bytes.
- Idempotent reruns should be safe.
- Keep manifest output deterministic.

`models.py`:

- Owns only modules with parameters/buffers.
- Keep construction order `(encoder, bottleneck, target_bottleneck,
  coarse_flow, decoder)` consistent with `train.py`.
- Do not put losses or optimizer logic here.

`losses.py`:

- Pure tensor math only.
- No `nn.Module`, no trainable parameters.
- Detach targets through `as_target()`.

`diagnostics.py`:

- Pure probes and optimizer-support utilities.
- Any new bypass/collapse test should return a flat `dict[str, float]`.
- Keep AGC and weight-decay exclusion predicates shared.

`train.py`:

- Orchestrates modules, losses, optimizer, EMA, logging, checkpoints.
- Keep train-step gradient paths readable.
- If a new loss is added, document exactly which modules it trains and add a
  test for the gradient contract.

`parse_logs.py` and `run_history.py`:

- Analysis helpers, not training dependencies.
- Do not couple core training to these scripts.

## 18. Current limitations to keep visible

- The project is currently coarse-latent only.
- The reconstruction `Decoder` reconstructs frozen encoder features, not pixels.
- There is no inference sampler for real future generation yet.
- There is no fine-flow shuffled-c proof yet.
- There is no frame-generator dependency proof yet.
- Some docs preserve older constants and planned-stage placeholders. Verify
  against code before using them in implementation.

The right agent behavior is to preserve the current working contracts while
making progress on the specific requested change, not to opportunistically
complete the whole research roadmap.
