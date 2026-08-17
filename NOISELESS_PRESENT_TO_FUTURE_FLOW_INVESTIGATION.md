# Noiseless Present-to-Future Flow Investigation

Date: 2026-08-03  
Scope: current Phase-1/Stage-1 HJEPA-VWM implementation only. Investigation and patch plan; no implementation or run was performed.

## 1. Executive conclusion

The current default task is conditional rectified flow from Gaussian noise to the detached future EMA-bottleneck latent. In live variable names,

`eps_c -> z_c -> u_c_hat`, conditioned separately on online `abstract`, with target `target_abstract`.

The correct minimal interpretation of “noiseless present-to-future flow” is option A/C: use a detached present bottleneck latent as the source endpoint and the detached future EMA-bottleneck latent as the target endpoint. Concretely:

- `x_0 = target_present = target_bottleneck(detailed)`
- `x_1 = target_abstract`
- `x_tau = (1 - tau_c) * target_present + tau_c * target_abstract`
- `u_tau = target_abstract - target_present`
- `F_c(x_tau, tau_c, abstract)`
- `predicted_endpoint = x_tau + (1 - tau_c) * u_c_hat`
- inference starts from a present bottleneck state, not from zeros or Gaussian noise.

This formulation is dimensionally compatible with the existing `CoarseFlow`: both endpoints are `(B,N_c,D_c)` and the separate condition is also `(B,N_c,D_c)`. It does not require a projection or a changed `F_c` input width. The source should be the **EMA/target bottleneck encoding of the present** (`target_present`), not the online `abstract`, because both endpoints then share one frozen coordinate system and remain stop-gradient targets. The online `abstract` should initially remain as the separate conditioning stream so that the experiment changes only the flow source. Its redundancy should be measured, not removed in the same experiment.

Option B (raw present encoder embedding to raw future encoder embedding) is incompatible with the current `F_c` interface: default encoder tensors are `(B,1024,1024)`, versus `(B,32,256)` for `F_c`. It would need token/width projections and would abandon the intended abstract bottleneck task. Option D (direct deterministic residual prediction) is technically distinct because it removes `tau_c` and interpolation. Option E (set `eps_c=0`) is not present-to-future flow: it creates a zero-to-future radial path and target velocity equal to the future latent, while providing no present source endpoint.

Recommendation: **go** to a gated implementation experiment only after the tests/metrics in Sections 7-10 are agreed. Do not combine the first experiment with a final velocity projection or changed condition dropout.

## 2. Current implementation trace

### 2.1 Normal Stage-1 call chain

Confirmed top-level path:

1. `train.py:2324` `main()` parses CLI/config and calls the training orchestration.
2. `train.py:1720` `run_training()` builds the dataloader/modules and, at `train.py:1887`, calls `train_step()`.
3. `train.py:366-463` `train_step()` moves `ClipBatch.context`/`.target` to the device, calls `_coarse_forward()`, samples flow state/time, calls `CoarseFlow.forward()`, and computes flow MSE.
4. `train.py:563` calls `loss.backward()` on the combined loss.

No `eval.py`, rollout sampler, ODE solver, or production future-generation inference path exists in the current repository. The only endpoint estimate is the one-step formula at `train.py:547-550` (training, only when `lambda_recon_pred > 0`) and `train.py:1391-1395` (diagnostic readout).

### 2.2 Present/future clip selection

| Step | Evidence | Input | Output | Next call |
|---|---|---|---|---|
| Choose window | `data.py:303-324`, `SSV2Dataset._window_indices` | `num_frames`; defaults `T=8`, `frame_stride=2`, `horizon_k=4` | `context_idx`, `target_idx`, each length 8. Context ends at `t`; target ends at `t+k`. Short clips repeat/clamp the last frame. | `SSV2Dataset.__getitem__` |
| Decode and shared transform | `data.py:350-377`, `SSV2Dataset.__getitem__` | selected indices | `ClipSample.context`, `.target`, each `(8,3,256,256)`, float32 raw `[0,1]` | `_collate_clip_samples` |
| Batch | `data.py:380-395`, `_collate_clip_samples` | list of samples | `ClipBatch.context`, `.target`: `(B,8,3,256,256)` | `train_step` via dataloader |
| Device seam | `train.py:424-430`, `train_step` | `ClipBatch` | `context_clip`, `target_clip` on device, same shape | `_coarse_forward` |

The same decoded stack receives the same resize/crop/color jitter before it is split (`data.py:367-376`), preserving geometric correspondence between the two windows.

### 2.3 Frozen encoder and bottleneck paths

`train.py:287-325` (`_coarse_forward`) is the definitive seam:

1. `detailed = encoder(context_clip)` under `torch.no_grad()` (`train.py:315-318`).
2. `abstract = bottleneck(detailed)` outside `no_grad` (`train.py:319`). This is the exact present condition passed to `F_c`; it remains gradient-connected to the online bottleneck.
3. `target_detailed = encoder(target_clip)` under `torch.no_grad()` (`train.py:320-323`).
4. `target_abstract = target_bottleneck(target_detailed)` (`train.py:324`). `TargetBottleneck.forward` runs its EMA bottleneck under `no_grad` and calls `as_target()` (`models.py:397-407`).

`FrozenEncoder.forward` validates `(B,8,3,256,256)`, normalizes once, invokes the selected frozen backend, validates the resolved token geometry, and returns `(B,N_e,D_e)` (`encoders.py:365-390`). The default `vjepa2_vitl16` registry entry is `D_e=1024`, layout `(4,16,16)` and therefore `N_e=1024` (`encoders.py:573-580`).

`Bottleneck.forward` maps detailed tokens to shared memory and initializes its abstract slots from the same learned `self.queries` table for every clip (`models.py:312-352`). Both the online bottleneck and its EMA copy use the same architecture and inherited query ordering (`models.py:237-253`, `358-407`). Default output is `(B,32,256)`.

Optional whitening occurs symmetrically on `detailed` and `target_detailed` at `train.py:317-323`; it does not change shape. Defaults keep whitening off.

### 2.4 Exact current flow construction

Default full-latent mode (`predict_residual=False`):

1. `flow_target = target_abstract` (`train.py:457`). This is the **full future EMA-bottleneck latent**, not an encoder embedding and not a residual.
2. `eps_c = torch.randn_like(target_abstract)` (`train.py:458`).
3. `tau_c = torch.rand(B, device=device)` (`train.py:459`), a `(B,)` uniform sample in `[0,1)`; `_broadcast_tau` expands it to `(B,1,1)` (`losses.py:31-35`).
4. `z_c = interpolate(flow_target, eps_c, tau_c)` (`train.py:460`), where `losses.py:53-64` implements

   `z_c = (1 - tau_c) * eps_c + tau_c * flow_target`.

5. `u_c = velocity_target(flow_target, eps_c)` (`train.py:461`), where `losses.py:67-76` implements

   `u_c = flow_target - eps_c`.

6. `u_c_hat = coarse_flow(z_c, tau_c, abstract)` (`train.py:462`).
7. `flow_loss = mean((u_c_hat - u_c)^2)` (`train.py:463`; `losses.py:105-114`).

Optional residual mode (`predict_residual=True`) does **not** implement the requested noiseless path. It computes detached

`target_present = target_bottleneck(detailed)` and `flow_target = target_abstract - target_present`, then samples scaled Gaussian `eps_c = std(flow_target) * randn_like(flow_target)` (`train.py:448-455`; `losses.py:79-102`). It therefore flows from scaled noise to a temporal residual.

### 2.5 Every `F_c` input and output

`CoarseFlow.forward` is at `models.py:525-560`.

- `z_c`: `(B,N_c,D_c)`, default `(64,32,256)`.
- `tau_c`: `(B,)`, default `(64,)`; sinusoidal `_timestep_embedding` produces `(B,D_c)`, and `time_mlp` preserves `(B,D_c)` (`models.py:462-479`, `517-519`, `557`).
- `abstract`: `(B,N_c,D_c)`, default `(64,32,256)`, the online present bottleneck latent.
- Training-only condition dropout: a `(B,)` Bernoulli mask with probability `condition_dropout=0.10` replaces the whole per-example `abstract` stream by learned `null_condition` (`models.py:541-548`; default at `config.py:88`).
- Before concatenation, `slot_pos + z_type` is added to `z_c`; `slot_pos + cond_type` is added to `abstract` (`models.py:549-556`).
- Concatenation order is **`[z_c, abstract]` along token dimension**, producing `(B,2*N_c,D_c)`, default `(64,64,256)` (`models.py:556`).
- Six default AdaLN transformer blocks process the combined stream; `self.norm(x[:, :N_c])` returns only the first stream (`models.py:558-560`).
- Output `u_c_hat`: `(B,N_c,D_c)`, default `(64,32,256)`.

There is no learned final linear velocity projection after `self.norm`; the normalized first stream is the velocity output.

### 2.6 Endpoint construction and losses

Current one-step endpoint estimate:

`endpoint = z_c + (1 - tau_c) * u_c_hat` (`train.py:547`, `1392`).

In full-latent mode, `c_hat = endpoint`. In residual mode, `c_hat = abstract + endpoint` (`train.py:548-550`, `1393-1395`). This is algebraically an estimate at time 1 under a constant-velocity step from the sampled `tau_c`; it is not a multi-step numerical integration path.

Losses in `train_step`:

- Always in predictive mode: `flow_loss` (`train.py:493`).
- Always added with default weight `0.10`: variance floor on online `abstract` (`train.py:471`, `494`; default `config.py:177`).
- Optional: covariance, slot-diversity, and SIGReg penalties on online `abstract` (`train.py:475-507`; defaults all zero at `config.py:186,201,205`).
- Optional present reconstruction: `decoder(abstract)` versus `detailed` (`train.py:522-534`; `lambda_recon=0` by default at `config.py:216`).
- Optional predicted-future endpoint reconstruction: `decoder(c_hat)` versus `target_detailed` (`train.py:535-562`; `lambda_recon_pred=0` by default at `config.py:241`). The default training path therefore does not reconstruct or penalize the endpoint directly.

Diagnostics disable condition dropout and report current model/copy/batch-mean velocity losses (`diagnostics.py:390-432`) plus `L_recon_present`, `L_recon_cplus`, `L_recon_chat`, shuffled-latent reconstruction, and video gap (`train.py:1351-1430`).

## 3. Current equations in clean notation

Use code names rather than generic architecture labels:

### Default full-future-latent flow

`abstract = B(E(context_clip))` (online, gradient-connected)  
`target_abstract = stopgrad(B_EMA(E(target_clip)))`  
`eps_c ~ Normal(0,I)`  
`tau_c ~ Uniform(0,1)` independently per example  
`z_c(tau_c) = (1-tau_c) eps_c + tau_c target_abstract`  
`u_c(tau_c) = target_abstract - eps_c`  
`u_c_hat = F_c(z_c, tau_c, abstract)`  
`L_flow = mean ||u_c_hat-u_c||^2`  
`endpoint_hat = z_c + (1-tau_c)u_c_hat`  
`c_hat = endpoint_hat`.

### Optional temporal-residual flow

`target_present = stopgrad(B_EMA(E(context_clip)))`  
`flow_target = target_abstract - target_present`  
`residual_sigma = std(flow_target)`  
`eps_c = residual_sigma * Normal(0,I)`  
`z_c = (1-tau_c)eps_c + tau_c flow_target`  
`u_c = flow_target-eps_c`  
`endpoint_hat = z_c+(1-tau_c)u_c_hat`  
`c_hat = abstract+endpoint_hat`.

## 4. Exact tensor-shape and gradient table

Default numerical values use `B=64`, `vjepa2_vitl16`, `N_e=4*16*16=1024`, `D_e=1024`, `N_c=32`, `D_c=256` (`config.py:30,61-63,135`; `encoders.py:573-580`).

| Code tensor | Meaning | Symbolic shape | Default shape | Gradient/detach status |
|---|---|---:|---:|---|
| `context_clip` | present raw clip | `(B,T,3,H,W)` | `(64,8,3,256,256)` | data, no grad |
| `target_clip` | future raw clip | `(B,T,3,H,W)` | `(64,8,3,256,256)` | data, no grad |
| `detailed` | present frozen encoder tokens (possibly whitened) | `(B,N_e,D_e)` | `(64,1024,1024)` | produced under `no_grad` |
| `target_detailed` | future frozen encoder tokens | `(B,N_e,D_e)` | `(64,1024,1024)` | produced under `no_grad` |
| `abstract` | online present bottleneck latent / condition | `(B,N_c,D_c)` | `(64,32,256)` | attached; gradients reach `B` through `F_c` and regularizers |
| `target_present` | EMA present bottleneck latent; residual mode only today | `(B,N_c,D_c)` | `(64,32,256)` | detached via `TargetBottleneck` |
| `target_abstract` | EMA future bottleneck latent | `(B,N_c,D_c)` | `(64,32,256)` | detached target |
| `flow_target` | full `target_abstract`, or detached temporal residual | `(B,N_c,D_c)` | `(64,32,256)` | detached |
| `eps_c` | Gaussian source, unit or residual-scaled | `(B,N_c,D_c)` | `(64,32,256)` | sampled, no grad |
| `tau_c` | scalar flow time per example | `(B,)` | `(64,)` | sampled, no grad; broadcast `(B,1,1)` |
| `z_c` | interpolated flow state | `(B,N_c,D_c)` | `(64,32,256)` | depends only on detached target/noise in current code |
| `cat([z_c,abstract])` | `F_c` token sequence | `(B,2N_c,D_c)` | `(64,64,256)` | condition half carries gradients to `B` |
| `u_c` | target velocity | `(B,N_c,D_c)` | `(64,32,256)` | target/no grad |
| `u_c_hat` | predicted velocity | `(B,N_c,D_c)` | `(64,32,256)` | gradients train `F_c` and condition path into `B` |
| `endpoint` | one-step predicted flow endpoint | `(B,N_c,D_c)` | `(64,32,256)` | grad only if prediction reconstruction is enabled |
| `c_hat` | predicted future latent | `(B,N_c,D_c)` | `(64,32,256)` | full endpoint, or online `abstract + endpoint` in residual mode |
| `decoder(c_hat)` | predicted future detailed features | `(B,N_e,D_e)` | `(64,1024,1024)` | executed in training only when `lambda_recon_pred>0` |

## 5. Candidate noiseless formulations

### A. Replace Gaussian noise with the present bottleneck representation

Recommended, with an important precision: source from `target_present = target_bottleneck(detailed)`, not directly from online `abstract`.

`x_0 = target_present`  
`x_1 = target_abstract`  
`x_tau = (1-tau_c)target_present + tau_c target_abstract`  
`u_tau = target_abstract-target_present`.

Dimensional compatibility: exact. No projection or `F_c` width change. Both endpoints are detached EMA-bottleneck coordinates. `abstract` remains the online condition.

Using online `abstract` as `x_0` is also shape-compatible, but it would introduce a new gradient path from the flow-state half of `F_c` into `B`, while the future endpoint remains in EMA coordinates. That changes both source semantics and optimization routing, confounding the first experiment.

### B. Present encoder embedding to future encoder embedding

`detailed -> target_detailed` is semantically direct but dimensionally incompatible with `F_c`: default `(B,1024,1024)` rather than `(B,32,256)`. It would require a new operator or projections/token compression, would be far more expensive, and would bypass the current abstract prediction claim. Not recommended for Task 3.

### C. Present compressed latent to future compressed latent

This is the code-grounded version of A and the recommendation. In this repository, “compressed latent” means bottleneck outputs. Use EMA present and EMA future endpoints for a stationary target geometry, while retaining online `abstract` as conditioning/training input.

### D. Direct deterministic residual prediction

Predict `target_abstract-target_present` directly from `abstract`, with no sampled `tau_c` and no `z_c` path. This needs a distinct forward interface/head or a convention for a dummy state/time. It is not flow matching and should be a separate baseline. It may be simpler and potentially stronger at one fixed horizon, but cannot establish whether continuous-time training/integration is useful.

### E. Set Gaussian noise to zero

This yields

`z_c = tau_c * target_abstract`, `u_c = target_abstract`, and inference source `0`.

It is dimensionally executable but scientifically degenerate for the stated hypothesis. Every path is a radial scaling from the origin, present information exists only in the separate condition, and the learned velocity can collapse toward target reconstruction conditioned on time. It neither defines present as the source endpoint nor tests temporal displacement. It also creates an artificial zero state that is not produced by the encoder/bottleneck.

## 6. Token correspondence and recommended formulation

### Confirmed alignment evidence

- Present and future clips use the same frozen encoder instance and resolved `FeatureLayout` (`train.py:315-324`; `encoders.py:365-390`).
- `Bottleneck` is applied identically to both clips; `TargetBottleneck` is a deep-copied EMA version of the online bottleneck (`models.py:358-407`).
- Each bottleneck pass initializes slot `j` from query row `self.queries[j]` (`models.py:282-288`, `345`). There is no per-sample permutation, Hungarian matching, or other slot-matching mechanism.
- `CoarseFlow` explicitly adds the same learned `slot_pos[j]` to source-state and conditioning slot `j`, with comments stating that this binds the two slot streams (`models.py:499-507`, `514-516`, `549-556`).
- Present and future compressed tensors match exactly at `(B,N_c,D_c)`.

### What the code does not prove

Shared query index gives a stable architectural correspondence, but it does not prove semantic identity of slot `j` across time. Cross-attention is content-dependent; a slot can attend to different objects/regions in the two clips, and there is no explicit temporal slot-consistency loss or matching. A tokenwise straight path is therefore **architecturally plausible, not empirically established**.

Required precondition diagnostic: measure per-slot present/future similarity and compare diagonal alignment against off-diagonal/permuted matching. Report at least diagonal cosine, best-match cosine, assignment consistency, and the gap between identity and optimal assignment. If identity is much worse than optimal assignment, straight tokenwise interpolation is poorly justified even though shapes match.

The source endpoint already contains present information. Retaining `abstract` as a separate condition duplicates present information through a second, online stream. This is intentional for the first controlled comparison; later zero/shuffle tests must be interpreted as testing reliance on the **explicit condition**, not reliance on the present overall.

### Recommended exact formulation

`source endpoint x_0 = target_present = target_bottleneck(detailed)`  
`target endpoint x_1 = target_abstract`  
`interpolation x_tau = (1-tau_c) * target_present + tau_c * target_abstract`  
`target velocity u_tau = target_abstract - target_present`  
`F_c inputs = (x_tau, tau_c, abstract)` with existing `[state, condition]` concatenation  
`predicted endpoint = x_tau + (1-tau_c) * F_c(x_tau,tau_c,abstract)`  
`inference starting state = target_bottleneck(detailed)` when using the trained EMA target bottleneck, or another explicitly chosen present bottleneck whose coordinate-system choice is validated. The cleanest training/inference-consistent first implementation uses the EMA bottleneck for the start state and reports the online-vs-EMA start gap.

An unresolved operational choice is whether production inference should use `B_EMA(E(context))` for both start and condition, or retain online `B(E(context))` as condition. Current checkpoints save both modules, so either is implementable, but training/evaluation must use the same convention it claims.

## 7. Minimum file-by-file patch plan (do not implement yet)

### `config.py`

- Add a typed field following existing dataclass conventions, preferably `flow_source: str = "noise"`, choices `{"noise","present"}`.
- Keep `noise` as the default for exact baseline reproducibility.
- Define interaction with `predict_residual`. Recommended first version: reject `flow_source="present"` together with `predict_residual=True`, because present-to-future flow already has displacement `target_abstract-target_present`; combining both would instead flow present-to-residual and be semantically unclear.

### `train.py`

- `parse_args()` (`train.py:1958+`): add `--flow-source {noise,present}`.
- `main()` (`train.py:2324+`): copy the CLI override into `cfg.train.flow_source`.
- `finalize_training_config()` (`train.py:2260+`): validate supported values and incompatible combinations.
- `train_step()` (`train.py:448-463`):
  - preserve current lines byte-for-byte under `flow_source == "noise"`;
  - under `present`, compute `target_present = target_bottleneck(detailed)`;
  - remove Gaussian sampling from this branch only;
  - set `flow_target = target_abstract`, `flow_source_tensor = target_present`;
  - call `interpolate(flow_target, flow_source_tensor, tau_c)` and `velocity_target(flow_target, flow_source_tensor)`.
- Rename generalized local/helper arguments from `eps_c` to `source_c` only if it improves clarity without obscuring baseline compatibility. A smaller first patch can keep helper parameter names stable but must not call present latents “noise” in docs/metrics.
- Endpoint formula remains `z_c + (1-tau_c)u_c_hat`; no mathematical change is required.
- `_run_diagnostics_impl()` (`train.py:1270-1303`) must mirror the selected source exactly and avoid noise draws in present mode.
- `reconstruction_readouts()` accepts a general interpolated state already; update names/docstrings, not behavior.
- Add explicit logged categorical/numeric provenance for the selected flow source and source/target/displacement norms.

### `losses.py`

- The current `interpolate(z_target, eps, tau)` and `velocity_target(z_target, eps)` math is already general for any same-shaped source. Prefer renaming `eps` to `source` and updating docstrings to state endpoint-agnostic semantics, while retaining equations.
- Add shape equality validation if the codebase's pure-loss helpers convention permits it.
- Do not add a second near-duplicate interpolation helper unless preserving the baseline's exact code path is judged more important than naming cleanliness.

### `models.py`

- No input dimension or concatenation change is required.
- Update `CoarseFlow` docstrings from “noised future” to “interpolated flow state.”
- Keep separate `abstract` conditioning for the first experiment.
- Do not add the proposed final velocity projection in this patch; it is a separate operator-capacity axis.

### `diagnostics.py`

- Generalize `coarse_baselines(..., eps_c, ...)` to a source tensor.
- For present-source full-latent mode, copy-present velocity is exactly the target velocity, so the current “copy velocity” baseline must be redefined carefully: the endpoint copy baseline is `c_hat_copy=target_present`, with endpoint MSE/cosine against `target_abstract`. Comparing velocity MSE to the oracle path velocity would incorrectly make copy look perfect if it uses the true displacement.
- Add endpoint-level metrics listed in Section 8. Keep historical metric keys for noise mode.
- Add condition zero/shuffle readouts as diagnostics, interpreted only as explicit-condition reliance.

### Checkpoint/provenance and docs

- Ensure `flow_source` is serialized through existing config/checkpoint/provenance machinery and checked on resume.
- If implemented, update `AGENT_FILES/AGENTS.md`, `GUIDES/PROBLEMS_METRICS_AND_EXPERIMENTS.md`, and relevant tests in the same change, as required by repository governance.

### Tests

- `tests/test_phase1_contract.py`: CLI/default/validation; default `noise` preserves current contract.
- New focused flow-source tests (or `tests/test_optimizer_and_flow.py`): exact source/interpolation/velocity equations; no `randn_like` call in present mode; shape equality; condition concatenation unchanged.
- Train-step test with fakes: `target_present` comes from `TargetBottleneck(detailed)`, both endpoints detached, and gradients still reach online `B` only through the explicit condition/regularizers (not through source/target).
- Endpoint reconstruction test: exact `tau=0`, `tau=1`, and perfect-velocity endpoint recovery.
- Diagnostic test: copy-present endpoint baseline and gain have correct values; condition shuffle/zero does not mutate RNG.
- Checkpoint/provenance round-trip and incompatible-resume tests.
- Slot-correspondence diagnostic tests using constructed identity/permuted slots.

## 8. Controlled experiment matrix

Hold dataset, sample order, transforms, encoder alias/revision, whitening choice, bottleneck/EMA, `F_c` width/depth, optimizer/LRs/schedule, batch size, horizon, decoder/reconstruction flags and weights, SigREG/variance/covariance/slot settings, condition dropout, seed, and training steps constant. Because current per-step RNG is `seed+step` (`train.py:396-401`), the present-source arm will necessarily consume fewer random draws if it removes `randn`; use dedicated generators or explicit per-purpose streams so `tau_c` and condition dropout remain matched across arms.

| Arm | Source/state | Target | Objective | Purpose |
|---|---|---|---|---|
| 1. Current noisy baseline | `eps_c ~ N(0,I)` | `target_abstract` | current rectified-flow velocity MSE | control |
| 2. Present-to-future | `target_present` | `target_abstract` | present/future straight-path velocity MSE | hypothesis |
| 3. Copy-present | `target_present` returned unchanged | `target_abstract` | no learned flow; endpoint metrics only | minimum useful baseline |
| 4. Direct residual | `abstract` input; predict `target_abstract-target_present` directly | `target_abstract` after add-back | deterministic residual MSE, no `tau` | tests whether flow formalism adds value |

Use identical initial trainable weights for learned arms and record initialization hash. The copy arm is evaluation-only on the same fixed validation batches.

### Required metrics

At training and fixed-diagnostic cadence as appropriate:

- velocity/flow MSE (`L_flow`, historical keys preserved);
- integrated endpoint MSE and endpoint cosine distance in abstract space;
- `L_recon_chat` versus frozen future encoder features, plus `L_recon_cplus` as decoder ceiling; this is the repository's implemented encoder-embedding-space readout;
- copy endpoint MSE/cosine and `gain_over_copy = copy_error-model_error` plus a ratio with a numerical floor;
- `||predicted_endpoint-target_present||`, `||target_abstract-target_present||`, and their ratio/correlation;
- prediction and copy metrics by `horizon_k` (separate controlled runs today because multi-horizon conditioning is not implemented);
- Euler/solver endpoint metrics using 1, 2, 4, and 8 evaluations from the same present start, with condition fixed. These require a new evaluation integrator; they are not current training behavior;
- current stability/representation metrics: `grad_norm`, `grad_skipped`, `instability_warn`, AGC metrics, `c_std_*`, `c_effective_rank`, `c_slot_*`, `c_plus_std_*`, `c_plus_effective_rank`, cross-video cosines, attention entropy, reconstruction video gap, and coarse baseline ratios where semantically valid;
- explicit-condition reliance: endpoint degradation when condition is zeroed or batch-shuffled, alongside the unchanged present source state;
- slot-alignment metrics described in Section 6.

Success requires statistically credible gain over copy at the same horizon, lower future endpoint/reconstruction error than noisy flow without worse collapse/stability, displacement norms that track true temporal change rather than explode/vanish, and no material degradation as integration steps increase.

## 9. Interaction with other flow-predictor concerns

### Keep separate from the first noiseless-flow comparison

- **Learned final velocity projection after final normalization:** do not include. It changes `F_c` expressivity/output geometry and could rescue or harm both sources independently. Run it as a later 2x2 (`noise/present` x `projection off/on`) only after the source effect is measured.
- **Condition dropout 10% -> 0%:** do not include. Hold the current `0.10` constant in the primary experiment. Then ablate dropout separately or run a 2x2. Changing it simultaneously would confound source information with condition availability.

### Must include as diagnostics, not as a training change

- **Does `F_c` ignore the present condition?** Include zero/shuffle condition readouts for every learned arm. In present-source mode, failure under shuffle measures reliance on the separate `abstract` stream only. Little degradation does **not** mean the model ignores the present, because `target_present` is already in its evolving state.

For a stronger decomposition, evaluate four inference probes without retraining: normal source/normal condition, normal source/shuffled condition, shuffled source/normal condition, and shuffled source/shuffled condition. The latter two deliberately test source reliance but must keep targets paired correctly and be labeled out-of-distribution probes.

## 10. Risks, unresolved questions, and go/no-go checklist

### Risks and unresolved questions

1. **Slot semantics:** shared query indices support correspondence but do not prove it. Identity-vs-optimal slot matching must be measured.
2. **EMA versus online inference state:** the recommended detached training source is `B_EMA(e_t)`, while the current explicit condition is online `B(e_t)`. Decide and document the inference convention.
3. **Redundant present paths:** the state and condition may make one another unnecessary. This is measurable and not inherently wrong.
4. **Deterministic straight paths:** for a single future paired with each present, the task may reduce to deterministic regression; direct residual prediction is essential as a baseline.
5. **Endpoint-only shortcut:** one-step reconstruction can look good while the vector field away from the sampled path is poor. Multi-step consistency and trajectory metrics are required.
6. **RNG confounding:** removing `randn_like` shifts the global RNG sequence unless streams are separated, changing `tau` and condition dropout despite a shared seed.
7. **Copy metric semantics:** velocity-space copy baselines from noise mode cannot be reused blindly in present-source mode; compare endpoints.
8. **Whitened runs:** source and target remain compatible only if both use the same fixed whitener, as current `_coarse_forward` does.
9. **No current sampler:** claims about 2/4/8-step inference remain proposed until an evaluation-only integrator is implemented and tested.

### Final go/no-go checklist

- [ ] `flow_source=noise` is default and reproduces the existing equations, RNG contract, metrics, and checkpoints.
- [ ] `flow_source=present` uses detached `target_present` and detached `target_abstract` from the same EMA bottleneck coordinate system.
- [ ] `predict_residual` interaction is explicitly rejected or precisely specified.
- [ ] No new gradient enters `B_EMA`; `as_target()` remains the named detach boundary.
- [ ] Existing `F_c` dimensions and `[state, condition]` order remain unchanged.
- [ ] Identity-vs-matched slot correspondence is measured before interpreting tokenwise interpolation as semantically valid.
- [ ] Endpoint copy baseline and gain-over-copy are implemented correctly.
- [ ] RNG streams keep `tau_c`, condition dropout, data order, and other stochastic mechanisms matched across arms.
- [ ] Endpoint MSE/cosine, displacement norms, encoder-feature reconstruction, and 1/2/4/8-step evaluation are tested.
- [ ] Condition shuffle/zero is interpreted as explicit-stream reliance, not total present reliance.
- [ ] Final velocity projection and condition-dropout changes are excluded from the primary comparison.
- [ ] Config, CLI, checkpoint/provenance, tests, and required living docs are updated together.
- [ ] No training is launched until the controlled matrix and acceptance thresholds are approved.

## Confirmed behavior versus inference

Everything attributed to file/line locations above is confirmed from the current repository. The recommendation that shared slots are sufficiently aligned for a straight path is **not confirmed**; it is a testable hypothesis. The recommended EMA-present source is a design conclusion derived from current tensor shapes and gradient boundaries, not behavior already implemented.
